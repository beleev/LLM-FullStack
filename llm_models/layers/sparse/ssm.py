"""
Selective State Space Model (S6 / Mamba 核心) — 非 Attention 分支

背景:
    Attention 是 O(T^2) 的序列建模器; 状态空间模型 (SSM) 是 O(T) 的替代方案。
    - S4 (Gu et al., 2022): 线性时不变 SSM, 参数与输入无关, 推理 O(1)/step
    - Mamba / S6 (Gu & Dao, 2023): **selective** SSM, 把 SSM 参数做成输入的函数,
      让模型可以"依据 token 内容决定记住/遗忘什么", 在 LM 任务上首次打平 Transformer

数学核心 (离散化 SSM):
    连续 SSM:  h'(t) = A·h(t) + B·x(t);   y(t) = C·h(t)
    离散化:    h_t = Ā·h_{t-1} + B̄·x_t;   y_t = C·h_t + D·x_t
    其中:      Ā = exp(Δ·A);  B̄ ≈ Δ·B  (零阶保持近似)
    "selective" = Δ, B, C 都是 x_t 的函数 (Mamba 创新),
                  A 保持为可学常数但也按 Δ 做时变离散化

本文件提供教学友好的 SelectiveSSM:
    - 纯 Python 顺序 scan (for-loop), 便于阅读; 生产用 CUDA 并行 scan
    - 通道 (d_model) 并行独立 SSM, 每通道一套 (A, B, C, D, Δ) 投影
    - 省略 1D conv branch (Mamba 真实 block 在 SSM 前还有一条 conv 分支),
      仅保留 SSM 核心, 让注意力 ↔ SSM 的对比更清晰

后续若要更贴 Mamba block, 可在外层 (models/mamba.py) 再加 conv + 门控分支。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Selective SSM 层 (教学版, 纯 Python scan)

    输入 [B, T, D], 每个通道独立跑一个状态维 d_state 的 SSM, 再汇总。

    参数设计:
        - A: [D, d_state] 学习到的对数稳定 (parameterize as -exp(A_log) 保证负实部 → 收敛)
        - D: [D] 直通残差 (skip connection in SSM, 相当于 y += D*x)
        - x_proj: 从 x 生成 Δ, B, C 三组参数 (selective: 随 x 变化)
        - dt_proj: 把低维 Δ 升到 [B, T, D] 每通道一个步长

    前向 (教学版 for-loop):
        for t in 1..T:
            Δ_t = softplus(linear(x_t))                # [B, D]
            A_bar = exp(Δ_t · A)                       # [B, D, d_state]
            B_bar = Δ_t · B_t                          # [B, D, d_state] (零阶保持近似)
            h_t = A_bar · h_{t-1} + B_bar · x_t        # [B, D, d_state]
            y_t = (h_t · C_t).sum(-1) + D · x_t        # [B, D]

    Args:
        d_model: 输入输出通道数
        d_state: 每通道 SSM 的隐状态维度 (Mamba 典型 16)
        dt_rank: Δ 的低秩投影维度, 默认 ceil(d_model / 16)
        dt_min/max: Δ 初始化范围 (softplus^{-1} 后的线性区间)
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

        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # --- 固定部分 (不随 x 变) ---
        # A 的负对数形式: 运行时 A = -exp(A_log), 保证负实部 → SSM 稳定收敛
        # 初值取 [1..d_state] 的对数, 模仿 HiPPO 初始化的衰减谱
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))           # [D, d_state]
        self.D = nn.Parameter(torch.ones(d_model))         # 每通道独立 skip 增益

        # --- selective 部分 (随 x 变) ---
        # 一次线性投影到 [Δ_low | B | C], 再拆开, 省一次 matmul
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        # Δ 从低秩 dt_rank 升到 D, 每通道一个步长
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Δ 初始化: 让 softplus(bias) ∈ [dt_min, dt_max], 位于 SSM 稳定的时间尺度
        # 先在 log 空间均匀采, 再 exp 回线性空间 → 得到对数均匀分布的 Δ 初值,
        # 然后求 softplus^{-1} 作为 bias, 保证 dt_low=0 时 softplus(bias)=dt_init。
        with torch.no_grad():
            # 对数均匀: log(dt) ~ U(log(dt_min), log(dt_max)), 故 dt ~ loguniform
            log_dt = torch.empty(d_model).uniform_(math.log(dt_min), math.log(dt_max))
            dt_init = torch.exp(log_dt)
            # softplus^{-1}(y) = log(exp(y) - 1); 数值稳定形式: y + log(1 - exp(-y))
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)
            # weight 轻度缩放 (Mamba 官方做法)
            nn.init.uniform_(self.dt_proj.weight, -1.0 / dt_rank**0.5, 1.0 / dt_rank**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            y: [B, T, D]
        """
        B, T, D = x.shape
        N = self.d_state

        # 1) 从 x 投出 (Δ_low, B, C); B, C 都是 [B, T, N] 每通道共享
        x_proj = self.x_proj(x)                                # [B, T, dt_rank + 2N]
        dt_low, B_param, C_param = torch.split(
            x_proj, [self.dt_rank, N, N], dim=-1
        )
        # Δ 升到每通道独立: [B, T, D], softplus 保证 > 0
        delta = F.softplus(self.dt_proj(dt_low))               # [B, T, D]

        # 2) 离散化 A, B (零阶保持近似)
        #    A = -exp(A_log) 保证负实部; Δ 按通道广播
        A = -torch.exp(self.A_log)                             # [D, N]
        # Ā: [B, T, D, N]; B̄: [B, T, D, N] (B_param 跨通道广播)
        delta_expand = delta.unsqueeze(-1)                     # [B, T, D, 1]
        A_bar = torch.exp(delta_expand * A)                    # [B, T, D, N]
        B_bar = delta_expand * B_param.unsqueeze(-2)           # [B, T, D, N]

        # 3) 顺序 scan (教学版; 生产版用并行 scan kernel)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            # h_t = Ā_t ⊙ h_{t-1} + B̄_t · x_t   (逐元素乘法, 不跨 token)
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)  # [B, D, N]
            # y_t = (h_t · C_t).sum(-1) + D · x_t
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)          # [B, D]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                      # [B, T, D]

        # 直通残差 (SSM 里的 D·x 项)
        y = y + self.D * x
        return y
