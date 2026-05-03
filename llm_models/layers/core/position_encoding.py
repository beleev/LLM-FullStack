"""
位置编码模块 — 给无序 attention 注入 "顺序" 信息

包含:
- SinPositionalEncoding:    正弦位置编码 (Vaswani et al., 2017 原始 Transformer)
- RotaryPositionalEncoding: 旋转位置编码 RoPE (Su et al., 2021; GPT-NeoX/LLaMA/Qwen 标配)
- MultimodalRotaryEmbedding: M-RoPE (Qwen2-VL, 2024) — 三轴 (时间/高/宽) 多模态扩展

演进与动机:
    Sinusoidal/Learnable (绝对位置, 加在 embedding 上)
        → RoPE (相对位置, 旋转 Q/K, 长度外推性更好, 无需额外参数)
            → M-RoPE (按维度切段, 各段独立 RoPE, 把 1D 顺序扩展到 (T,H,W) 三轴)

为什么 RoPE 能编码相对位置:
    把每两维当成复数 z = a + ib，按位置 m 旋转角 mθ → z·e^{imθ}。
    Q_m 与 K_n 的内积 ⟨z_m, z_n⟩ = ⟨z, z⟩·e^{i(m-n)θ}，
    实部仅依赖 (m - n)，因此天然编码了 **相对** 距离，且无需任何可学参数。
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


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


class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE) — Su et al., 2021

    相对前代 (Sinusoidal/Learnable) 的关键优势:
        - 编码的是 **相对** 距离 (Q·K 内积只依赖位置差)
        - 不引入额外可学参数
        - 长度外推性更好 (NTK / YaRN 等微调即可扩到 128k)
        - 应用在 Q/K 而非 embedding 上，与 KV cache 兼容性好

    使用场景:
        - 输入既可是 [B, T, d_head] (旧接口)
        - 也可是 [B, H, T, d_head] (多头 4D 张量)
        - position_ids 可显式传入以支持 KV-cache / M-RoPE / 自定义位置

    Args:
        d_head: 单头维度 (注意不是 d_model；RoPE 作用在 per-head 维度上)
        max_len: 预计算的最大位置 (上线后超长序列要么扩 max_len 要么用 NTK 缩放)
        base: 频率基数 (默认 10000；扩长用 NTK-aware 时会调到 1e6 量级)
    """

    def __init__(self, d_head: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()

        if d_head % 2 != 0:
            raise ValueError(f"RoPE 要求 d_head 为偶数，当前 {d_head}")

        self.d_head = d_head
        self.max_len = max_len

        # 不同维度对有不同旋转频率: 低维快 (近距离敏感)，高维慢 (远距离敏感)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_len, d_head//2]
        # 把 freqs 复制一份拼接 → [max_len, d_head]，对应 _rotate_half 的前后两半排布
        emb = torch.cat((freqs, freqs), dim=-1)

        # 预计算 cos/sin 表，运行时直接索引；不进 state_dict (persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

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
        把 head_dim 按比例切成 (temporal, height, width) 三段，
        每段独立 RoPE，独立位置索引：
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

        # 每段单独一个 RoPE，互不干扰
        self.axis_ropes = nn.ModuleList(
            [RotaryPositionalEncoding(d, max_len=max_len, base=base) for d in section_dims]
        )

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

        # 把 head_dim 切成三段 (T/H/W)，每段用对应轴的 position_ids 独立 RoPE，再拼回
        splits = torch.split(x, list(self.section_dims), dim=-1)
        rotated = [
            rope(seg, position_ids=position_ids[i])
            for i, (seg, rope) in enumerate(zip(splits, self.axis_ropes))
        ]
        return torch.cat(rotated, dim=-1)
