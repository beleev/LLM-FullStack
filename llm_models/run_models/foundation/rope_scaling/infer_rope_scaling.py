#!/usr/bin/env python
"""
RoPE 长度外推: plain / NTK / YaRN — 不训练, 直接看 "旋转角" (确定性, < 1 s)

设定: 训练长度 L = 2048, 目标长度 4L = 8192 (s = 4), d_head = 64 → 32 个频率 θ_i。
外推为什么坏: 第 i 维在位置 m 的旋转角是 m·θ_i。低频维在 L 内转不满一圈 (r_i = L·θ_i/2π < 1),
    模型只见过 [0, L·θ_i] 这一小段弧; 位置 > L 时角度落到 **从未见过的弧段** 上。
    高频维早就转了几百圈, 所有角度都见过, 不存在这个问题。
验证 (assert):
    1. plain: 最低频维在 4L 处的角度 = 训练时最大角度的 4 倍 (越界)
    2. NTK:   最低频恰好 ÷s (回到训练范围内), 最高频不变; 中间各维按 s^{-2i/(d-2)} 平滑过渡 (仍轻微越界)
    3. YaRN:  r_i > 32 的维原封不动 (保住局部分辨率), r_i < 1 的维整个 ÷s, 中间线性过渡; 温度 0.1·ln s + 1
    4. 线性内插 (PI, 所有维 ÷s) 作对照: 越界问题也解决了, 但最高频被压 4 倍 → 相邻 token 更难区分
    5. 缩放后仍是合法 RoPE: ⟨q_m, k_n⟩ 只依赖 m-n
"""

import math

import torch

from llm_models.layers.core.position_encoding import RotaryPositionalEncoding, scaled_inv_freq


def main():
    d_head, L, s = 64, 2048, 4.0
    plain, _ = scaled_inv_freq(d_head)
    ntk, _ = scaled_inv_freq(d_head, scaling="ntk", factor=s, original_max_len=L)
    yarn, mscale = scaled_inv_freq(d_head, scaling="yarn", factor=s, original_max_len=L)
    pi = plain / s                                                # Position Interpolation (Chen et al., 2023)

    rotations = L * plain / (2 * math.pi)                         # r_i: 训练长度内转了几圈
    seen = L * plain                                              # 训练时每维见过的最大角度
    print(f"d_head={d_head}, 训练长度 L={L}, 目标 {int(s)}L={int(s * L)}")
    print(f"{'维 i':>4} {'圈数 r_i':>10} | 位置 4L 处的角度 ÷ 训练见过的最大角度 (r_i<1 的维上 >1 即越界; r_i≥1 的维整圈都见过, 无所谓)")
    print(f"{'':>4} {'':>10} | {'plain':>7} {'PI':>7} {'NTK':>7} {'YaRN':>7}")
    for i in (0, 8, 16, 20, 24, 28, 31):
        ratio = [float(s * L * f[i] / seen[i]) for f in (plain, pi, ntk, yarn)]
        print(f"{i:>4} {float(rotations[i]):>10.2f} | " + " ".join(f"{r:>7.2f}" for r in ratio))

    low = rotations < 1                                           # 没转满一圈的维: 外推的病根
    high = rotations > 32
    assert low.any() and high.any()

    # 1) plain: 低频维越界 s 倍
    assert torch.allclose(s * L * plain[low] / seen[low], torch.full((int(low.sum()),), s))
    # 2) NTK: 两端精确, 中间单调
    assert torch.isclose(ntk[0], plain[0]) and torch.isclose(ntk[-1], plain[-1] / s)
    assert ((ntk / plain).diff() < 0).all()
    # 3) YaRN: 分段
    assert torch.equal(yarn[high], plain[high])
    assert torch.allclose(yarn[low], plain[low] / s)
    assert ((yarn / plain >= 1 / s - 1e-6) & (yarn / plain <= 1)).all()
    assert abs(mscale - (0.1 * math.log(s) + 1)) < 1e-9
    # 4) PI 把最高频也压了 s 倍: 相邻位置的角度差从 1 rad 降到 0.25 rad; YaRN / NTK 保住了
    assert float(pi[0]) == 0.25 and float(yarn[0]) == 1.0 and float(ntk[0]) == 1.0
    ntk_over = s * L * ntk[low] / seen[low]
    assert float(ntk_over.max()) > 1.2 and torch.isclose(ntk_over[-1], torch.tensor(1.0))
    print(f"\n低频维 (r<1) {int(low.sum())} 个: plain 越界 {s:.0f}×; YaRN/PI 全部拉回 1.00×; "
          f"NTK 只有最低频恰好 1.00×, 其余仍越界至多 {float(ntk_over.max()):.2f}× (所以 NTK 的实际可用长度 < s·L)")
    print(f"高频维 (r>32) {int(high.sum())} 个: YaRN 原封不动; PI 把相邻 token 角度差 1.00 → {float(pi[0]):.2f} rad")
    print(f"YaRN 温度补偿 mscale = 0.1·ln({s:.0f}) + 1 = {mscale:.4f}  (logits × mscale² = {mscale**2:.4f})")

    # 5) 相对位置性质: 同一对 (q, k), 位置 (m, n) 与 (m+Δ, n+Δ) 的内积相同
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, d_head), torch.randn(1, 1, d_head)
    for scaling in (None, "ntk", "yarn"):
        rope = RotaryPositionalEncoding(d_head, max_len=int(s * L), scaling=scaling, factor=s)
        dot = lambda m, n: float(                                    # noqa: E731
            (rope(q, position_ids=torch.tensor([m])) * rope(k, position_ids=torch.tensor([n]))).sum()
        )
        a, b = dot(100, 40), dot(100 + 5000, 40 + 5000)
        assert abs(a - b) < 1e-3 * max(1.0, abs(a)), (scaling, a, b)
    print("plain / NTK / YaRN 均保持 ⟨q_m, k_n⟩ 只依赖 m-n")


if __name__ == "__main__":
    main()
