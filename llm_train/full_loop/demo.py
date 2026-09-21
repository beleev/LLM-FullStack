"""
Full Loop — 把前面的模块拼成一个真的训练主循环

每个 step 依次发生 (括号里是出处):
    lr = warmup_cosine(step)                                   (m10)
    fp16 计算副本 ← all-gather(各 rank 的 fp32 master 分片)      (m05 ZeRO, m06 master weights)
    每 rank: K 个 micro-batch, fp16 前向/反向, loss × scale, 梯度累积   (m01, m02, m06)
    任一 rank 梯度含 Inf/NaN → 整步跳过, scale 减半             (m06 LossScaler, m10 NaN guard)
    reduce-scatter(mean) 梯度 → 每 rank 只拿自己那片            (m05, m09)
    全局范数裁剪: ‖g‖² 由各分片的平方和 all-reduce 得到          (m10)
    每 rank 在自己的 master/m/v 分片上做 Adam                   (m05)
    每 rank 各写一个 checkpoint 分片文件; 可从中途逐位续训        (m08)
读代码盯住: `train()` 是唯一的训练函数 —— world=1, micro=1, amp=False 就是单卡基线, 所以等价性断言是同一份代码的自洽检验。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from llm_train.core import (
    LinearModel, ToyDataStream, adam_update, all_gather, all_reduce_sum, banner, comm, flatten_tree,
    has_overflow, kv, load_checkpoint, max_abs_diff, reduce_scatter_sum, save_checkpoint, unflatten_like,
)
from llm_train.m06_mixed_precision.demo import LossScaler
from llm_train.m10_training_stability.demo import warmup_cosine_lr

F16, F32 = np.float16, np.float32
D_IN, D_OUT = 6, 2                                        # 6·2 + 2 = 14 个参数, 能被 world=2 整除
TOTAL_STEPS, WARMUP, BASE_LR, MAX_NORM = 40, 5, 0.05, 1.0
BAD_BATCH_STEP = 12                                       # 这一步的数据里混入一个 NaN (模拟坏样本)


def init_state(world: int) -> dict:
    flat = flatten_tree(LinearModel.init(D_IN, D_OUT, seed=0).params()).astype(F32)       # [14]
    ranks = [{"master": s.copy(), "m": np.zeros_like(s), "v": np.zeros_like(s)} for s in np.split(flat, world)]
    return {
        "step": 0, "opt_steps": 0, "skipped": 0,
        "ranks": ranks,                                   # 每 rank: fp32 master/m/v 各 [14/world]
        "scaler": LossScaler(init_scale=2.0**18, growth_interval=8),     # 起点故意偏高; 8 是玩具值
        "data": ToyDataStream(D_IN, D_OUT, batch_size=16, seed=42),
    }


def train_step(state: dict, micro: int, amp: bool) -> float:
    ranks, scaler, world = state["ranks"], state["scaler"], len(state["ranks"])
    dt = F16 if amp else F32
    like = LinearModel.init(D_IN, D_OUT).params()
    lr = warmup_cosine_lr(state["step"], TOTAL_STEPS, BASE_LR, WARMUP, min_ratio=0.1)

    x, y = state["data"].next_batch()                     # 全局 batch [16, 6]
    if state["step"] == BAD_BATCH_STEP:
        x[0, 0] = np.nan

    # 1) 计算副本: 把各 rank 的 master 分片转成低精度再 all-gather (通信的是 fp16, 省一半带宽)
    flat_lp = all_gather([rk["master"].astype(dt) for rk in ranks])      # world × [14]
    scale = scaler.scale if amp else 1.0

    # 2) 每 rank: micro-batch 累积。梯度以 "放大后的 fp16" 形式产生, 转 fp32 再累加
    local, losses = [], []
    for r in range(world):
        p = unflatten_like(flat_lp[r], like)
        model = LinearModel(p["W"], p["b"])
        xr, yr = np.split(x, world)[r], np.split(y, world)[r]            # [16/world, ...]
        acc = np.zeros(flat_lp[r].size, dtype=F32)
        for xb, yb in zip(np.split(xr, micro), np.split(yr, micro)):
            with np.errstate(over="ignore", invalid="ignore"):           # 溢出是预期内的事, 交给 scaler 处理
                loss, g = model.loss_and_grads(xb.astype(dt), yb.astype(dt), loss_scale=scale)
                acc += flatten_tree(g).astype(F32) / micro               # 等长 micro → 简单平均 (m01)
            losses.append(loss)
        local.append(acc)

    # 3) 溢出检查要 "全局一致": 任何一个 rank 坏了, 所有 rank 一起跳过 (真实实现: all-reduce 一个 found_inf 标志)
    overflow = has_overflow(local)
    if amp:
        scaler.update(overflow)
    state["step"] += 1
    if overflow:
        state["skipped"] += 1
        return float("nan")

    # 4) reduce-scatter: world × [14] → 每 rank [14/world]; 顺手 unscale 并取 rank 平均
    shards = [g / F32(scale * world) for g in reduce_scatter_sum(local)]

    # 5) 全局范数裁剪 —— 范数必须是 "所有分片" 的, 每 rank 只贡献自己的平方和
    sq = all_reduce_sum([np.array([np.sum(s.astype(np.float64) ** 2)]) for s in shards])[0]
    clip = min(1.0, MAX_NORM / (float(np.sqrt(sq[0])) + 1e-12))

    # 6) 分片 Adam: 每 rank 只碰自己的 master/m/v
    state["opt_steps"] += 1
    for rk, g in zip(ranks, shards):
        adam_update(rk["master"], g * F32(clip), rk["m"], rk["v"], state["opt_steps"], lr)
    return float(np.mean(losses))


def train(world: int, micro: int, amp: bool, steps: int, state: dict | None = None) -> dict:
    state = state or init_state(world)
    for _ in range(steps):
        state["last_loss"] = train_step(state, micro, amp)
    return state


def full_master(state: dict) -> np.ndarray:
    return np.concatenate([rk["master"] for rk in state["ranks"]])


def save_sharded(ckpt_dir: Path, state: dict) -> None:
    """每 rank 一个文件 (只含自己的分片) + 一个全局 meta —— 真实分布式 checkpoint 的最小形态。"""
    for r, rk in enumerate(state["ranks"]):
        save_checkpoint(ckpt_dir / f"rank{r}.pkl", rk)
    save_checkpoint(ckpt_dir / "meta.pkl", {
        "world": len(state["ranks"]),
        "step": state["step"], "opt_steps": state["opt_steps"], "skipped": state["skipped"],
        "scaler": state["scaler"].state_dict(), "data": state["data"].state_dict(),
    })


def load_sharded(ckpt_dir: Path) -> dict:
    meta = load_checkpoint(ckpt_dir / "meta.pkl")
    state = init_state(meta["world"])                                    # "新进程": 先建空壳, 再逐项覆盖
    state["ranks"] = [load_checkpoint(ckpt_dir / f"rank{r}.pkl") for r in range(meta["world"])]
    state["scaler"].load_state_dict(meta["scaler"])
    state["data"].load_state_dict(meta["data"])
    state.update({k: meta[k] for k in ("step", "opt_steps", "skipped")})
    return state


def val_loss(flat: np.ndarray, data: ToyDataStream) -> float:
    rs = np.random.RandomState(99)
    x = rs.randn(64, D_IN).astype(F32)
    p = unflatten_like(flat, LinearModel.init(D_IN, D_OUT).params())
    return float(np.mean((x @ p["W"] + p["b"] - (x @ data.true_W + data.true_b)) ** 2))


def main() -> None:
    banner("Full Loop - Mini Distributed Training")

    # ---- A) 全 fp32: DP(2) × 累积(2) × 分片 Adam == 单卡 ----
    single32 = train(world=1, micro=1, amp=False, steps=TOTAL_STEPS)
    dist32 = train(world=2, micro=2, amp=False, steps=TOTAL_STEPS)
    diff32 = max_abs_diff(full_master(single32), full_master(dist32))
    print("\n[A] fp32: 分布式 vs 单卡")
    kv("max |Δ master|", f"{diff32:.1e}  (只差 float32 求和顺序)")
    kv("NaN batch 被跳过的步数 (两边)", f"{single32['skipped']} / {dist32['skipped']}")
    assert diff32 < 1e-5 and single32["skipped"] == dist32["skipped"] == 1

    # ---- B) AMP: fp16 计算 + fp32 master + 动态 loss scale ----
    comm.reset()
    dist16 = train(world=2, micro=2, amp=True, steps=TOTAL_STEPS)
    wire = comm.summary()
    single16 = train(world=1, micro=1, amp=True, steps=TOTAL_STEPS)
    start = val_loss(full_master(init_state(1)), dist16["data"])
    v16, v32 = val_loss(full_master(dist16), dist16["data"]), val_loss(full_master(dist32), dist32["data"])
    moved = max_abs_diff(full_master(dist16), full_master(init_state(1)))
    print("\n[B] AMP (fp16 + master + LossScaler)")
    kv("跳过的步数 (溢出 + NaN batch)", f"{dist16['skipped']} / {TOTAL_STEPS},  最终 scale = 2^{int(np.log2(dist16['scaler'].scale))}")
    kv("val loss: 初始 → AMP / fp32", f"{start:.4f} → {v16:.5f} / {v32:.5f}")
    kv("max |Δ master| AMP 分布式 vs AMP 单卡", f"{max_abs_diff(full_master(dist16), full_master(single16)):.1e}  (参数共移动 {moved:.2f})")
    kv(f"通信 ({TOTAL_STEPS} 步合计)", wire)
    assert dist16["skipped"] >= 2 and dist16["skipped"] == single16["skipped"], "起始 scale 过高必然溢出; NaN batch 也必须被拦下"
    assert np.isfinite(full_master(dist16)).all()
    assert v16 < 0.05 * start and abs(v16 - v32) < 0.02 * start, "AMP 的终点与 fp32 基本重合"
    assert max_abs_diff(full_master(dist16), full_master(single16)) < 0.02 * moved, "只差 fp16 舍入"

    # ---- C) 分片 checkpoint → 新进程恢复 → 逐位续训 ----
    with tempfile.TemporaryDirectory(prefix="llm_train_full_") as tmp:
        half = train(world=2, micro=2, amp=True, steps=TOTAL_STEPS // 2)
        save_sharded(Path(tmp), half)
        files = sorted(p.name for p in Path(tmp).iterdir())
        resumed = train(world=2, micro=2, amp=True, steps=TOTAL_STEPS // 2, state=load_sharded(Path(tmp)))
    print("\n[C] checkpoint / resume")
    kv("文件", files)
    kv("max |Δ master| 续训 vs 不中断", f"{max_abs_diff(full_master(resumed), full_master(dist16)):.1e}")
    assert np.array_equal(full_master(resumed), full_master(dist16)), "续训必须逐位相同"
    assert all(np.array_equal(a[k], b[k]) for a, b in zip(resumed["ranks"], dist16["ranks"]) for k in ("m", "v"))
    assert resumed["scaler"].state_dict() == dist16["scaler"].state_dict()
    assert resumed["skipped"] == dist16["skipped"]

    print("\n  OK: DP × 累积 × AMP × ZeRO 分片 Adam × 裁剪 × NaN guard × 分片 checkpoint, 全部有断言兜底。")


if __name__ == "__main__":
    main()
