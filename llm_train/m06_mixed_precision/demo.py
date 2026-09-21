"""
M06 — 混合精度: FP16 / BF16 + loss scaling + FP32 master weights

是什么: 矩阵乘用 16 位算 (省一半显存和带宽, 吃满 Tensor Core), 参数更新留在 FP32。
解决的瓶颈: 显存 + 算力。引入的新问题全是数值问题:
    FP16 (5 位指数, 10 位尾数): 最小 ~6e-8 → 小梯度下溢成 0;  最大 65504 → 大值溢出成 Inf
    BF16 (8 位指数,  7 位尾数): 范围同 FP32, 不需要 loss scaling; 代价是精度 (1 附近步长 2^-7 vs 2^-10)
关键机制: grad_fp16 = backward(loss × S);  若含 Inf/NaN → 跳过本步, S /= 2;  否则 grad / S 后更新 FP32 master。
读代码盯住: `LossScaler.scale` 的轨迹, 以及被跳过的 step 上 master 权重一位都不动。
说明: numpy 没有 bfloat16, 用 core.fake_quant_float 把尾数截到 7 位来模拟。
"""
from __future__ import annotations

import numpy as np

from llm_train.core import LinearModel, ToyDataStream, banner, fake_quant_float, has_overflow, kv

F16, F32 = np.float16, np.float32


class LossScaler:
    """动态 loss scale: 溢出就减半并跳过; 连续 growth_interval 个好 step 就翻倍试探上限。

    PyTorch GradScaler 默认 init=2^16, growth_interval=2000。demo 里传 2~10 只是为了
    在几十步内看到 "增长 → 再溢出 → 回退" 的完整循环, 真实训练别用这么小的值。
    """

    def __init__(self, init_scale: float = 2.0**16, growth_interval: int = 2000):
        self.scale = float(init_scale)
        self.growth_interval = growth_interval
        self.good_steps = 0

    def update(self, overflow: bool) -> None:
        if overflow:
            self.scale = max(1.0, self.scale / 2)
            self.good_steps = 0
            return
        self.good_steps += 1
        if self.good_steps >= self.growth_interval:
            self.scale *= 2
            self.good_steps = 0

    def state_dict(self) -> dict:                      # scale 也是训练状态, 要进 checkpoint (full_loop 用)
        return {"scale": self.scale, "good_steps": self.good_steps}

    def load_state_dict(self, state: dict) -> None:
        self.scale, self.good_steps = float(state["scale"]), int(state["good_steps"])


def main() -> None:
    banner("M06 - Mixed Precision (FP16 / BF16)")

    # ---- 1) FP16 vs BF16: 范围 vs 精度 ----
    print("\n[1] 同样 16 位, 指数/尾数分法不同")
    probes = np.array([1e-8, 70000.0, 1 + 2**-9], dtype=np.float64)
    with np.errstate(over="ignore"):
        as_fp16 = probes.astype(F16).astype(np.float64)
    as_bf16 = fake_quant_float(probes, "bf16")
    for name, a, b in zip(["1e-8 (小梯度)", "70000 (大激活)", "1+2^-9 (小更新)"], as_fp16, as_bf16):
        kv(name, f"fp16 → {float(a):<14.10g} bf16 → {float(b):.10g}")
    assert as_fp16[0] == 0 and as_bf16[0] > 0, "FP16 下溢, BF16 不下溢"
    assert np.isinf(as_fp16[1]) and np.isfinite(as_bf16[1]), "FP16 上溢, BF16 不上溢"
    assert as_fp16[2] != 1.0 and as_bf16[2] == 1.0, "BF16 尾数少 3 位: 小增量被吃掉"

    # ---- 2) loss scaling 把小梯度搬进 FP16 的可表示区 ----
    print("\n[2] loss scaling")
    tiny = np.array([1e-8, 3e-8, 1e-7], dtype=F32)
    S = 2.0**15
    recovered = (tiny * S).astype(F16).astype(F32) / S            # 放大 → 存成 fp16 → 转回 fp32 再缩小
    kv("直接转 fp16", tiny.astype(F16).tolist())
    kv(f"×{int(S)} → fp16 → ÷{int(S)}", recovered.tolist())
    direct_err = np.abs(tiny.astype(F16).astype(F32) - tiny) / tiny
    kv("直接转的相对误差", [f"{e:.0%}" for e in direct_err])
    assert tiny.astype(F16)[0] == 0 and direct_err.min() > 0.15, "fp16 最小 subnormal ≈ 6e-8: 更小的变 0, 附近的误差巨大"
    assert np.allclose(recovered, tiny, rtol=0.01)

    # ---- 3) 为什么要 FP32 master: fp16 在 1.0 附近步长 ~1e-3, 小于它的更新全部丢失 ----
    print("\n[3] FP32 master weights")
    w16_only = F16(1.0)
    master = F32(1.0)
    for _ in range(100):
        w16_only = F16(w16_only - F16(1e-5))                       # 纯 fp16 训练: 每步都被舍回 1.0
        master = F32(master - F32(1e-5))                           # master 每步都在动
        if _ == 0:
            first_copy = F16(master)
    kv("纯 fp16 权重, 100 步后", float(w16_only))
    kv("fp32 master, 100 步后", f"{float(master):.6f}")
    kv("master 的 fp16 副本: 1 步后 / 100 步后", f"{float(first_copy)} / {float(F16(master))}")
    assert w16_only == 1.0, "更新 1e-5 << fp16 步长, 100 步白训"
    assert first_copy == 1.0 and master != 1.0, "单步看 fp16 副本没变, 但 master 记住了"
    assert F16(master) < 1.0, "累计够一个 fp16 步长后, 副本才跟着动"

    # ---- 4) 真实的 fp16 反向溢出 → 动态 scaler 跳步回退 ----
    print("\n[4] 动态 loss scaler (fp16 前向/反向真实溢出, 非手工构造 Inf)")
    data = ToyDataStream(d_in=8, d_out=4, batch_size=16, seed=6)
    model = LinearModel.init(8, 4, seed=6)                         # W, b 即 fp32 master
    scaler = LossScaler(init_scale=2.0**20, growth_interval=5)     # 故意起得过高; 5 是玩具值
    history, skipped, lr = [], 0, 0.3
    for step in range(24):
        x, y = data.next_batch()
        half = LinearModel(model.W.astype(F16), model.b.astype(F16))          # fp16 计算副本
        with np.errstate(over="ignore", invalid="ignore"):
            loss, g16 = half.loss_and_grads(x.astype(F16), y.astype(F16), loss_scale=scaler.scale)
        overflow = has_overflow(g16)                               # Inf 来自 fp16 乘法本身超过 65504
        before = model.W.copy()
        if overflow:
            skipped += 1                                           # 跳过: master 不能被 Inf 污染
            assert np.array_equal(before, model.W)
        else:
            model.apply_grads({k: g.astype(F32) / scaler.scale for k, g in g16.items()}, lr)
        history.append((int(np.log2(scaler.scale)), overflow, loss))
        scaler.update(overflow)
    print("    log2(scale): " + " ".join(f"{s}{'!' if o else ''}" for s, o, _ in history) + "   (! = 溢出并跳过)")
    kv("跳过的 step", f"{skipped} / 24")
    kv("loss (首个有效步 → 末步)", f"{next(l for _, o, l in history if not o):.4f} → {history[-1][2]:.4f}")
    assert history[0][1], "scale=2^20 时第一步必然溢出"
    assert 0 < skipped < 12
    assert any(o for _, o, _ in history[8:]), "增长后会再次探到上限并回退 —— scale 在上限附近振荡是正常的"
    assert history[-1][2] < 0.5 * next(l for _, o, l in history if not o)
    assert np.isfinite(model.W).all()

    print("\n  OK: FP16 需要 loss scaling + skip-step + FP32 master; BF16 用精度换范围, 省掉 scaling (现代训练默认)。")


if __name__ == "__main__":
    main()
