"""
M08 — Checkpoint / Resume (断点续训)

是什么: 把 "下一步计算会读到的一切状态" 落盘, 使得 挂掉重启 == 从未挂过 (逐位相同)。
解决的瓶颈: 容错。千卡任务 MTBF 以小时计, 不能精确续训就无法复现 loss 曲线、无法排查 spike。
状态清单: 参数 · 优化器状态 (动量/Adam m,v) · step · 数据流 (seed + cursor) · RNG 状态 (dropout / shuffle)。
读代码盯住: `train_steps` 每步从 `rng` 抽 dropout mask —— RNG 不恢复, 续训的 mask 序列就变了。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from llm_train.core import (
    LinearModel, MomentumSGD, ToyDataStream, banner, kv, load_checkpoint, max_abs_diff, save_checkpoint,
)

DROPOUT_P = 0.25


def train_steps(model, optim, data, rng: np.random.RandomState, steps: int) -> None:
    for _ in range(steps):
        x, y = data.next_batch()
        mask = (rng.rand(*x.shape) >= DROPOUT_P) / (1 - DROPOUT_P)     # 输入 dropout: 每步消耗 RNG
        _, grads = model.loss_and_grads((x * mask).astype(np.float32), y)
        optim.step(model.params(), grads)


def fresh(model_seed=8, data_seed=9, rng_seed=10):
    model = LinearModel.init(4, 2, seed=model_seed)
    return model, MomentumSGD(model.params(), lr=0.05, momentum=0.8), \
        ToyDataStream(4, 2, batch_size=6, seed=data_seed), np.random.RandomState(rng_seed)


def snapshot(model, optim, data, rng, step) -> dict:
    return {
        "model": {k: v.copy() for k, v in model.params().items()},
        "optim": optim.state_dict(),
        "data": data.state_dict(),                 # seed + cursor
        "rng": rng.get_state(),                    # MT19937 的 624 个字 + 位置
        "step": step,
    }


def restore(payload, model, optim, data, rng, skip=()) -> int:
    """skip 用来做消融: 故意漏恢复某一项, 看续训会不会偏。"""
    model.load_params(payload["model"])
    if "optim" not in skip:
        optim.load_state_dict(payload["optim"])
    if "data" not in skip:
        data.load_state_dict(payload["data"])
    if "rng" not in skip:
        rng.set_state(payload["rng"])
    return payload["step"]


def main() -> None:
    banner("M08 - Checkpoint / Resume")

    ref = fresh()
    train_steps(*ref, steps=5)                                          # 参照: 不中断连跑 5 步

    run = fresh()
    train_steps(*run, steps=3)
    with tempfile.TemporaryDirectory(prefix="llm_train_ckpt_") as tmp:
        path = Path(tmp) / "step_0003.pkl"
        save_checkpoint(path, snapshot(*run, step=3))                   # 原子写: tmp → rename
        size = path.stat().st_size

        results = {}
        for skip in [(), ("rng",), ("optim",), ("data",)]:
            # "新进程": 所有对象用不同的种子/超参重建, 只有 checkpoint 能把它们拉回正轨
            model, _, data, rng = fresh(model_seed=999, data_seed=777, rng_seed=555)
            optim = MomentumSGD(model.params(), lr=0.05, momentum=0.8)
            step = restore(load_checkpoint(path), model, optim, data, rng, skip=skip)
            train_steps(model, optim, data, rng, steps=5 - step)
            results[skip] = max_abs_diff(ref[0].params(), model.params())

    kv("checkpoint 大小", f"{size} B (其中 RNG 状态 ~2.5KB, 比模型还大)")
    kv("完整恢复: max |Δ| vs 不中断", f"{results[()]:.1e}")
    kv("漏掉 RNG 状态", f"{results[('rng',)]:.1e}")
    kv("漏掉 optimizer 状态", f"{results[('optim',)]:.1e}")
    kv("漏掉 data seed/cursor", f"{results[('data',)]:.1e}")

    assert results[()] == 0.0, "完整恢复必须逐位相同"
    assert all(results[s] > 1e-6 for s in results if s), "任何一项漏恢复都会让续训偏离"
    print("\n  OK: 参数 + 优化器 + 数据游标 + RNG 四样齐全才逐位续上; 漏任何一样都不报错, 只是悄悄偏了。")


if __name__ == "__main__":
    main()
