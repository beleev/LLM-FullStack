"""
位置编码 — 给置换不变的 attention 注入 "顺序"

是什么: SinPositionalEncoding (2017, 绝对位置, 加在 embedding 上)
        → RotaryPositionalEncoding / RoPE (2021, 旋转 Q/K, 内积只依赖相对距离, 零参数)
        → NTK / YaRN 缩放 (2023, 不重训或少量微调把上下文扩 4~32×)
        → MultimodalRotaryEmbedding / M-RoPE (Qwen2-VL, head_dim 切三段分别编码 T/H/W)
RoPE 核心: 每两维看成复数 z, 位置 m 处乘 e^{imθ_i}, θ_i = base^{-2i/d};
           ⟨q_m, k_n⟩ 的实部只含 (m-n)θ_i → 天然相对位置。
外推为什么坏: 低频维 (θ_i 小) 在训练长度 L 内转不满一圈, 位置 > L 时出现 **从未见过的角度**。
    NTK:  base ← base·s^{d/(d-2)}, 最低频恰好 ÷s, 最高频不变
    YaRN: 按 "训练长度内转了几圈 r_i" 分段: r<1 整个 ÷s (内插), r>32 不动 (外推), 中间线性过渡;
          再把 logits 乘 (0.1·ln s + 1)² 补偿长序列 softmax 变平
读代码时盯住: `inv_freq` —— 所有缩放方法都只是在改这一个向量。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def sinusoidal_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    通用正弦频率嵌入 — Transformer 位置编码与扩散 timestep 嵌入共用的数学核心

    把任意位置张量 (整数 token 序号 / 连续 diffusion timestep / 帧号 / ...) 映射到
    [..., dim] 维频率特征向量。频率族:  1 / max_period^(2i/dim), i ∈ [0, dim/2)。

    复用关系:
        - SinPositionalEncoding 把它用在整数位置 [0, max_len)
        - TimestepEmbedding (diffusion/adaln.py) 把它用在连续扩散时间 t
        二者共享同一频率族，差异只是位置取值与排布顺序。

    Args:
        positions:   任意形状的位置张量 (int 或 float)，输出形状为 positions.shape + (dim,)
        dim:         输出特征维度 (必须为偶数)
        max_period:  最低频率周期 (Transformer 与 DDPM 都默认 10000)
        interleaved: 输出排布
            - True  → [sin_0, cos_0, sin_1, cos_1, ...] (Transformer 原始风格)
            - False → [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}] (DDPM/DiT 风格)
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_embedding 要求 dim 为偶数，当前 {dim}")

    half = dim // 2
    # freqs[i] = 1 / max_period^(i / half) = 1 / max_period^(2i/dim)，与原 SinPE 公式等价
    freqs = torch.exp(
        -torch.arange(half, device=positions.device, dtype=torch.float)
        * (math.log(max_period) / half)
    )
    # [..., 1] * [half] → [..., half]
    args = positions.float().unsqueeze(-1) * freqs

    if interleaved:
        out = torch.empty(*positions.shape, dim, device=positions.device, dtype=torch.float)
        out[..., 0::2] = torch.sin(args)
        out[..., 1::2] = torch.cos(args)
        return out
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class SinPositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding) — 原始 Transformer 用法

    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    设计动机:
        不同维度选用不同频率 (10000^(2i/d) 等比衰减)，构成 "位置指纹"，
        理论上模型可线性组合 sin/cos 表示任意相对偏移 (因和差化积)。
        缺点: 加在 embedding 上属于绝对位置，长度外推效果一般，已基本被 RoPE 取代。

    特点:
        - 位置编码直接加到 embedding 上
        - 用 register_buffer 注册：随 .to(device) 迁移，但不参与梯度
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 复用通用 sinusoidal_embedding: 把整数位置 [0, max_len) 映射为频率特征
        # interleaved=True 保留 Transformer 原始 sin/cos 交替排布
        positions = torch.arange(0, max_len, dtype=torch.float)
        pe = sinusoidal_embedding(positions, d_model, interleaved=True)  # [max_len, d_model]

        # persistent=False: 不写入 state_dict，避免下游 checkpoint 体积膨胀
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # offset: KV cache 解码时, 新 token 的绝对位置从 offset 开始
        return x + self.pe[:, offset : offset + x.size(1)]    # [B, T, D] + [1, T, D]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] (前后各一半) -> [-x2, x1]

    这是 GPT-NeoX/HuggingFace 风格的 RoPE 排布 (前一半 vs 后一半成对)，
    与论文中的 (相邻两元素成对) 数学等价但实现上对硬件更友好。
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    应用 RoPE 旋转公式: x' = x * cos + rotate_half(x) * sin

    数学背景:
        把 x 的相邻两维当成复数 z = a + ib，乘以 e^{iθ} = cos θ + i sin θ:
            z · e^{iθ} = (a cos θ - b sin θ) + i (a sin θ + b cos θ)
        实数张量上等价于:  x * cos + rotate_half(x) * sin

    Args:
        x: [..., T, D_rope]
        cos, sin: 可广播到 x 的形状，最后两个维度需覆盖 (T, D_rope)
    """
    return (x * cos) + (_rotate_half(x) * sin)


def scaled_inv_freq(
    d_head: int,
    base: float = 10000.0,
    scaling: Optional[str] = None,
    factor: float = 1.0,
    original_max_len: int = 2048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    返回 (inv_freq [d_head/2], attention 温度补偿 mscale)。

    scaling=None:   θ_i = base^{-2i/d}
    scaling="ntk":  base' = base · s^{d/(d-2)}   → θ_0 不变, θ_last 恰好 ÷ s
    scaling="yarn": r_i = L·θ_i / 2π (训练长度内转的圈数)
                    γ_i = clip((r_i - β_slow) / (β_fast - β_slow), 0, 1)
                    θ_i' = (1-γ_i)·θ_i/s + γ_i·θ_i;   mscale = 0.1·ln(s) + 1
    """
    exponent = torch.arange(0, d_head, 2).float() / d_head              # 2i/d  [d_head/2]
    if scaling is None or factor == 1.0:
        return 1.0 / base**exponent, 1.0
    if scaling == "ntk":
        return 1.0 / (base * factor ** (d_head / (d_head - 2))) ** exponent, 1.0
    if scaling == "yarn":
        inv_freq = 1.0 / base**exponent
        rotations = original_max_len * inv_freq / (2 * math.pi)          # r_i
        gamma = ((rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0, 1)
        # γ=1 (高频, 转了很多圈, 管局部顺序): 原样外推; γ=0 (低频, 没转满一圈): 内插 ÷s
        return (1 - gamma) * inv_freq / factor + gamma * inv_freq, 0.1 * math.log(factor) + 1.0
    raise ValueError(f"未知 RoPE scaling: {scaling!r} (可选 None / 'ntk' / 'yarn')")


class RotaryPositionalEncoding(nn.Module):
    """
    RoPE (Su et al., 2021), 可选 NTK / YaRN 长度外推。

    输入 [B, T, d_head] 或 [B, H, T, d_head]; position_ids 显式传入以支持 KV cache / M-RoPE。

    Args:
        d_head:  单头维度 (不是 d_model)
        max_len: 预计算 cos/sin 表的长度 (扩展后的目标长度)
        base:    频率基数
        scaling / factor / original_max_len: 见 scaled_inv_freq;
                 original_max_len 默认 max_len / factor (即 "训练时的长度")
    """

    def __init__(
        self,
        d_head: int,
        max_len: int = 5000,
        base: float = 10000.0,
        scaling: Optional[str] = None,
        factor: float = 1.0,
        original_max_len: Optional[int] = None,
    ):
        super().__init__()

        if d_head % 2 != 0:
            raise ValueError(f"RoPE 要求 d_head 为偶数，当前 {d_head}")

        self.d_head = d_head
        self.max_len = max_len

        if original_max_len is None:
            original_max_len = int(max_len / factor)
        # 低维转得快 (近距离敏感), 高维转得慢 (远距离敏感)
        inv_freq, self.mscale = scaled_inv_freq(
            d_head, base, scaling, factor, original_max_len
        )
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)    # [max_len, d_head/2]
        emb = torch.cat((freqs, freqs), dim=-1)                         # [max_len, d_head] 对应 _rotate_half 的前后两半

        # cos/sin 同乘 mscale ⇒ q·k 乘 mscale² = YaRN 的 1/t; 不进 state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", emb.cos() * self.mscale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * self.mscale, persistent=False)

    def _lookup(
        self, seq_len: int, position_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 默认按 0..seq_len-1 取；KV cache / M-RoPE 场景需显式传 position_ids
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        return cos, sin

    def _broadcast(
        self, cos: torch.Tensor, sin: torch.Tensor, target_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 cos/sin 扩到 target_dim 维以便与 x 广播。"""
        while cos.dim() < target_dim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_head] 或 [B, H, T, d_head]
            position_ids: None / [T] / [B, T]；None 表示用 arange(0, T)

        Returns:
            旋转后的张量，形状与 x 一致
        """
        seq_len = x.size(-2)
        cos, sin = self._lookup(seq_len, position_ids)

        # 形状对齐: cos/sin 要能广播到 x。x 是 4D (多头) 时需要在 head 维插 1
        if x.dim() == 4:
            # x: [B, H, T, D]，目标 cos/sin: [B, 1, T, D] 或 [1, 1, T, D]
            if cos.dim() == 2:  # [T, D] -> [1, 1, T, D]
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif cos.dim() == 3:  # [B, T, D] -> [B, 1, T, D]
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        else:
            cos, sin = self._broadcast(cos, sin, x.dim())

        return apply_rotary_pos_emb(x, cos, sin)


class MultimodalRotaryEmbedding(nn.Module):
    """
    M-RoPE (Multimodal Rotary Position Embedding) — Qwen2-VL (2024) 核心创新

    解决的问题:
        视觉 patch 是 2D 网格 (帧 × 行 × 列)，强行展平成 1D 序列再用普通 RoPE
        会丢失空间结构。M-RoPE 让位置编码本身就有 (T, H, W) 三轴语义。

    核心思想:
        三轴共用同一条 RoPE 频率轴, 把 d_head/2 个频率按比例分给 (temporal, height, width),
        每段用各自轴的位置索引去旋转：
            - 文本 token：三轴位置索引相同 (退化为普通 1D RoPE)
            - 视觉 patch：三轴分别填 (帧号, 行号, 列号)
        这样文本和视觉能在同一个 attention 里直接交互，又各自保留时空先验。

    Args:
        d_head: 单头维度
        section_dims: (t_dim, h_dim, w_dim)，三段各占的维度；不传则近似三等分，
            且保证每段为偶数。三者之和必须等于 d_head。
        max_len: 每个轴的最大位置
        base: 频率基数
    """

    def __init__(
        self,
        d_head: int,
        section_dims: Optional[Tuple[int, int, int]] = None,
        max_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()

        if d_head % 2 != 0:
            raise ValueError(f"M-RoPE 要求 d_head 为偶数，当前 {d_head}")

        if section_dims is None:
            # 近似三等分，且每段为偶数 (RoPE 要求每段维度为偶数才能成对旋转)
            # 先算后两段 third (向下取偶)，剩余给第一段，避免凑不齐总维度
            third = (d_head // 3 // 2) * 2
            section_dims = (d_head - 2 * third, third, third)

        if sum(section_dims) != d_head:
            raise ValueError(
                f"section_dims {section_dims} 之和必须等于 d_head {d_head}"
            )
        if any(s % 2 != 0 for s in section_dims):
            raise ValueError(f"每段必须为偶数，当前 {section_dims}")

        self.d_head = d_head
        self.section_dims = tuple(section_dims)

        # 与 Qwen2-VL 原版一致: 三轴共用 **同一条** 频率轴, 只是把 d_head/2 个频率按段分给 T/H/W。
        # 这样三轴位置相同时 (纯文本) 与 1D RoPE 逐位相等; 若每段各建一个 RoPE (各自从最高频起算) 则不等。
        self.rope = RotaryPositionalEncoding(d_head, max_len=max_len, base=base)
        freq_axis = torch.repeat_interleave(
            torch.arange(3), torch.tensor([s // 2 for s in section_dims])
        )  # [d_head/2] 第 j 个频率归哪个轴
        # cos/sin 表是 cat(freqs, freqs) 排布, 轴归属同样复制一份 -> [d_head]
        self.register_buffer("dim_axis", torch.cat([freq_axis, freq_axis]), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_head] 或 [B, H, T, d_head]
            position_ids: [3, B, T]，三行分别是 (temporal, height, width) 位置

        Returns:
            旋转后的张量，形状与 x 一致
        """
        if position_ids.dim() != 3 or position_ids.size(0) != 3:
            raise ValueError(
                f"M-RoPE 的 position_ids 必须是 [3, B, T]，当前 {tuple(position_ids.shape)}"
            )

        cos = self.rope.cos_cached[position_ids]  # [3, B, T, d_head] 三个轴各查一次表
        sin = self.rope.sin_cached[position_ids]
        # 每个维度只取 "自己所属轴" 的那份 cos/sin -> [B, T, d_head]
        pick = self.dim_axis.view(1, 1, 1, -1).expand(1, *cos.shape[1:])
        cos = cos.gather(0, pick).squeeze(0)
        sin = sin.gather(0, pick).squeeze(0)
        if x.dim() == 4:  # [B, H, T, d_head]: 在 head 维广播
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return apply_rotary_pos_emb(x, cos, sin)


if __name__ == "__main__":
    # 自检: 三轴位置相同时 M-RoPE 必须与 1D RoPE 逐位相等; 只改 H 轴只应影响 H 段的维度
    torch.manual_seed(0)
    m = MultimodalRotaryEmbedding(16, (4, 6, 6), max_len=64)
    x = torch.randn(2, 3, 5, 16)
    pos = torch.arange(5).expand(2, 5)
    same = pos.expand(3, 2, 5)
    assert torch.equal(m(x, same), m.rope(x, position_ids=pos))
    moved = same.clone(); moved[1] += 7
    changed = (m(x, moved) != m(x, same)).any(dim=0).any(dim=0).any(dim=0)  # [d_head]
    assert torch.equal(changed, m.dim_axis == 1), changed
    print("M-RoPE 自检通过: 文本退化为 1D RoPE; H 轴位移只动", int(changed.sum()), "个维度")
