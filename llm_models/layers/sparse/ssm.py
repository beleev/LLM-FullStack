"""
Selective SSM (S6) — Mamba 的核心, 不用 attention 的序列建模

是什么: 每个通道维护一个 N 维隐状态 h, 逐 token 递推; 训练 O(T), 推理每步 O(1)。
解决了什么: attention 要回看全部历史 (O(T²) 计算 + 随 T 增长的 KV cache);
           S4 是线性时不变的, 无法按内容选择记忆。S6 让 Δ, B, C 成为 x_t 的函数 → "selective"。
公式:  h_t = Ā_t ⊙ h_{t-1} + B̄_t · x_t,   y_t = C_t · h_t + D · x_t
       Ā_t = exp(Δ_t · A),  B̄_t = Δ_t · B_t,  A = -exp(A_log) < 0,  Δ_t = softplus(·) > 0
数值稳定的全部秘密: A<0 且 Δ>0 ⇒ Ā ∈ (0,1), 状态只会衰减不会爆炸。
       (把 A 的负号去掉, T≈70 步就溢出成 inf/NaN — 见 run_models/.../mamba/readme.md)
读代码时盯住: h (唯一的"记忆", 形状 [B, D, N], 与序列长度无关)。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Args:
        d_model: 通道数 D (每个通道一个独立 SSM)
        d_state: 每通道隐状态维度 N (Mamba 典型 16)
        dt_rank: Δ 的低秩瓶颈, 默认 ceil(D / 16)
        dt_min/dt_max: 初始 Δ 的范围 (对数均匀)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        if dt_rank is None:
            dt_rank = max(1, math.ceil(d_model / 16))
        self.d_model, self.d_state, self.dt_rank = d_model, d_state, dt_rank
        self.dt_min, self.dt_max = dt_min, dt_max

        # A = -exp(A_log): 无论 A_log 学成什么, A 恒负。初值 1..N 模仿 HiPPO 的多时间尺度衰减谱
        A = torch.arange(1, d_state + 1, dtype=torch.float).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))                # [D, N]
        self.D = nn.Parameter(torch.ones(d_model))             # skip 增益

        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)  # x -> [Δ_low | B | C]
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)                # Δ_low -> 每通道 Δ
        self.reset_dt()

    @torch.no_grad()
    def reset_dt(self):
        """Δ 的专用初始化: 让 softplus(bias) 对数均匀落在 [dt_min, dt_max]。

        通用的 init_weights 会把 Linear bias 清零 (→ Δ≈0.69, 所有通道同一时间尺度),
        所以外层模型调完 init_weights 后必须再调一次本方法。
        """
        dt = torch.exp(torch.empty(self.d_model).uniform_(math.log(self.dt_min), math.log(self.dt_max)))
        self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # softplus⁻¹(dt)
        bound = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -bound, bound)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]。给了 cache 就从 cache["h"] 续跑并写回 (O(1) 解码)。"""
        B, T, D = x.shape
        N = self.d_state

        dt_low, B_t, C_t = self.x_proj(x).split([self.dt_rank, N, N], dim=-1)  # [B,T,r] [B,T,N] [B,T,N]
        delta = F.softplus(self.dt_proj(dt_low)).unsqueeze(-1)                 # [B, T, D, 1], > 0

        A = -torch.exp(self.A_log)                             # [D, N], < 0
        A_bar = torch.exp(delta * A)                           # [B, T, D, N], ∈ (0, 1)
        Bx = delta * B_t.unsqueeze(2) * x.unsqueeze(-1)        # [B, T, D, N]  = B̄_t · x_t

        h = cache["h"] if cache and "h" in cache else x.new_zeros(B, D, N)
        ys = []
        for t in range(T):  # 教学版顺序 scan; 生产用并行 scan kernel
            h = A_bar[:, t] * h + Bx[:, t]                                     # [B, D, N]
            ys.append((h * C_t[:, t].unsqueeze(1)).sum(-1))                    # [B, D]
        if cache is not None:
            cache["h"] = h
        return torch.stack(ys, dim=1) + self.D * x                             # [B, T, D]
