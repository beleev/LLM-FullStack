"""
adaLN-Zero — DiT 把 "时间步 t + 条件 c" 注入每一层的方式 (Peebles & Xie, 2023)

解决的问题: 扩散的条件是全局的 (每个 patch 收到同一份), 用 cross-attention 太贵;
adaLN 只把条件映射成 per-channel 的 (shift β, scale γ, gate α), FiLM 式整体调制。

    (β1, γ1, α1, β2, γ2, α2) = Linear(c)                      # c: [B, c_dim]
    x = x + α1 · Attn((1 + γ1) · LN(x) + β1)
    x = x + α2 · FFN ((1 + γ2) · LN(x) + β2)

"-Zero": 那个 Linear 的 weight/bias 初始化为 0 → α=0 → 每个 block 起步是恒等映射,
深层网络不会被随机残差污染。读代码时盯住: ada_modulation 与它的零初始化。
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from llm_models.layers.core.position_encoding import sinusoidal_embedding


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """FiLM 调制 (1 + scale)·x + shift。"+1" 让 scale=0 (零初始化) 时退化为恒等。"""
    # x: [B, T, D];  shift/scale: [B, D] → 插入 T 维后广播
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """
    时间步 t → sinusoidal 特征 → 两层 MLP, 得到条件向量 [B, d_model]。

    复用 core/position_encoding.py::sinusoidal_embedding (同一频率族 1/max_period^(2i/d)),
    只是 "位置" 从 token 序号换成扩散时间。

    量纲约定: t ∈ [0, 1000) (DDPM 整数步, 或 Flow Matching 的 t·1000)。
    频率族是为 "位置跨度上千" 设计的, 直接喂 t∈[0,1] 时大多数频率几乎不动,
    t=0.1 与 t=0.9 的特征余弦相似度 0.98; 缩放由 FlowMatchingScheduler / EulerFlowSampler 负责。
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
    PreLNBlock 的 "条件化" 版本: norm 输出先被 (γ, β) 调制, 子层输出再乘 gate α 进残差。

        x ──┬── adaLN(γ1, β1) ── Attn ──·α1──⊕──┬── adaLN(γ2, β2) ── FFN ──·α2──⊕── out
            └────────────────────────────────┘  └────────────────────────────────┘

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
        # c [B, c_dim] → [B, 6D] → 6 × [B, D]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada_modulation(c).chunk(6, dim=-1)

        # 1) 自注意力子层: adaLN(γ, β) 调制 → Attn → gate α 缩放 → 残差
        h = modulate(self.norm1(x), shift_a, scale_a)
        h = self.attn(q=h, k=h, v=h, mask=attn_mask)                  # [B, T, D]
        x = x + gate_a.unsqueeze(1) * h

        # 2) FFN 子层: 同样的结构
        h = modulate(self.norm2(x), shift_f, scale_f)
        h = self.ffn(h)
        x = x + gate_f.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    """
    DiT 最后一层: adaLN + Linear, [B, T, D] → [B, T, patch_out_dim], 之后由模型 unpatchify。

    输出 Linear 也零初始化 → 未训练的 DiT 输出恒为 0 → 初始 MSE = E[target²]
    (DDPM ≈ 1, Flow Matching ≈ 2), train 脚本据此做 sanity assert。

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
        shift, scale = self.ada_modulation(c).chunk(2, dim=-1)       # 2 × [B, D]
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
