"""
Adaptive LayerNorm - Zero (adaLN-Zero) — DiT 的条件注入基石

论文出处:
    "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)
    扩散 Transformer 中"如何把时间步 t 与条件 c 塞进每一层"的最佳答案。

为什么不是 cross-attention?
    扩散模型里的条件 (timestep t, class label / text embedding) 是 **全局** 的,
    每个 patch token 都接收同一份条件。cross-attn 参数量与算力消耗大,
    而 adaLN 只需把条件映射为 per-channel 的 (shift, scale) 二元组,
    像 FiLM 那样做"整体调制"即可, 算力几乎为零且效果更好 (DiT 论文实证)。

数学形式 (per block):
    c_emb = MLP(t_emb + cond_emb)                          # [B, D]
    (γ_attn, β_attn, α_attn, γ_ffn, β_ffn, α_ffn) = split(Linear(c_emb))
    h = x + α_attn · Attn( (1 + γ_attn) · LN(x) + β_attn )
    h = h + α_ffn  · FFN( (1 + γ_ffn ) · LN(h) + β_ffn  )

adaLN-Zero 关键: 把 **Linear 的权重初始化为 0**
    初始 γ=β=α=0 → 每个 block 起点都是恒等映射 (x 直通)。
    训练中 block 从"透明"慢慢学出调制强度, 是 DiT 训练稳定的关键。

对比 adaLN (不置零):
    初始 α≠0 会让残差路径立即被 Attn/FFN 的随机输出污染, 深网络会爆炸或塌陷。
    adaLN-Zero 从恒等映射起步, 让"是否激活某一层"成为可学习的 soft gate。
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.position_encoding import sinusoidal_embedding


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    FiLM 风格调制: (1 + scale) · x + shift

    为什么是 1 + scale 而不是 scale?
        加 1 让 scale=0 时退化为恒等映射 (x 不变), 配合 adaLN-Zero 初始化 scale=0,
        整块就变成 "LN(x)", 与 Pre-LN Block 前向完全一致。
    """
    # x: [B, T, D];  shift/scale: [B, D] → 插入 T 维后广播
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """
    扩散时间步 t 的 sinusoidal 嵌入 + 两层 MLP

    本质就是 "把扩散时间 t 当作连续位置喂给 sinusoidal positional encoding"，
    因此直接复用 core/position_encoding.py::sinusoidal_embedding，与
    SinPositionalEncoding 共享同一频率族 (1 / max_period^(2i/d))，
    只是位置取值从整数 token 序号换成连续 timestep。

    Args:
        d_model: 输出维度 (通常与模型主干同维)
        max_period: 控制最大频率, DDPM 原论文用 10000
    """

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model 必须为偶数, 当前 {d_model}")

        self.d_model = d_model
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def _sin_embed(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] 连续或离散 timestep → [B, d_model] 频率特征 (DDPM/DiT 风格 [cos|sin] 排布)"""
        return sinusoidal_embedding(
            t, self.d_model, max_period=float(self.max_period), interleaved=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sin_embed(t))


class AdaLNZeroBlock(nn.Module):
    """
    DiT Block with adaLN-Zero 条件注入

    结构:
        x ──┬── adaLN(γ1, β1) ── Attn ──·α1──┐
            │                                  ⊕ ──┬── adaLN(γ2, β2) ── FFN ──·α2──┐
            └────────────────────────────── ──┘    │                                ⊕ ── out
                                                    └───────────────────────────── ──┘

    条件张量 c (来自 TimestepEmbedding, 可选再加 class/text embedding) 经一次 Linear
    切成 6 段 (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)。
    adaLN-Zero 要求这个 Linear 的权重 + bias **初始化为 0**, 让初始 γ=β=α=0 → 恒等。

    与 PreLNBlock 的关系:
        可以看成 PreLNBlock 的 "条件化" 版本:
          - 把 norm1 的输出用 (1+γ1)·x + β1 调制后喂给 Attn
          - 把 Attn 的输出乘 α1 再残差
        没有条件时退化为 PreLNBlock 的等价物。

    Args:
        d_model:   模型维度
        c_dim:     条件向量 c 的维度 (通常等于 d_model)
        attn:      self-attention 模块
        ffn:       feed-forward 模块
        norm_cls:  LN 构造器, 默认 nn.LayerNorm (affine=False, 因为 scale/shift 由 adaLN 提供)
    """

    def __init__(
        self,
        d_model: int,
        c_dim: int,
        attn: nn.Module,
        ffn: nn.Module,
        norm_cls: Callable[[int], nn.Module] = lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6),
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)

        # 6 段调制参数: shift/scale/gate × (attn / ffn)
        self.ada_modulation = nn.Linear(c_dim, 6 * d_model, bias=True)

        # adaLN-Zero 初始化: 全部置零, 保证训练起点每个 block 都是恒等映射
        nn.init.zeros_(self.ada_modulation.weight)
        nn.init.zeros_(self.ada_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, T, D] patch token 序列
            c:         [B, c_dim] 条件嵌入 (timestep + 可选 class/text)
            attn_mask: 注意力掩码 (DiT 中通常是 None, 图像 patch 全部互见)
        """
        # 条件 → 6 段调制
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada_modulation(c).chunk(6, dim=-1)

        # 1) 自注意力子层: adaLN(γ, β) 调制 → Attn → gate α 缩放 → 残差
        h = modulate(self.norm1(x), shift_a, scale_a)
        h = self.attn(q=h, k=h, v=h, mask=attn_mask)
        x = x + gate_a.unsqueeze(1) * h

        # 2) FFN 子层: 同样的结构
        h = modulate(self.norm2(x), shift_f, scale_f)
        h = self.ffn(h)
        x = x + gate_f.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    """
    DiT 最后一层: adaLN + Linear 到输出通道

    把最后一个 block 的 [B, T, D] 映射到 [B, T, patch_out_dim], 之后由
    外部的 unpatchify 逻辑还原为 [B, C, H, W]。

    为什么不直接用普通 LN + Linear?
        保持 "每个 block (含终点) 都由 (shift, scale) 门控" 的一致性, 训练更稳;
        初始 shift=scale=0 让 FinalLayer 从"等价于 LN(x)"起步。

    Args:
        d_model:        输入维度
        c_dim:          条件维度
        patch_out_dim:  每 patch 要输出的通道数 (e.g. patch_size^2 * C)
    """

    def __init__(self, d_model: int, c_dim: int, patch_out_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ada_modulation = nn.Linear(c_dim, 2 * d_model, bias=True)
        self.linear = nn.Linear(d_model, patch_out_dim, bias=True)

        # adaLN-Zero 风格: 调制归零, 最终层输出投影也归零 (DiT 原论文做法)
        nn.init.zeros_(self.ada_modulation.weight)
        nn.init.zeros_(self.ada_modulation.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
