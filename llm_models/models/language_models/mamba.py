"""
Mamba 模型模块

论文出处:
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    (Gu & Dao, 2023)
    "Transformers are SSMs / Mamba-2" (Dao & Gu, 2024)

在本库演进地图中的位置:
    - 与 Transformer 并列的 **另一条主线**: 非注意力 / 线性复杂度
    - 用 Selective State Space Model (S6) 替代 self-attention
    - 复杂度 O(T) (vs attention 的 O(T^2)), 长上下文推理 5× 快 (论文)

Mamba block 设计 (与 Transformer block 的对照):
    Transformer:  x -> LN -> Attn     -> Add ; x -> LN -> FFN -> Add
    Mamba:        x -> LN -> MambaLayer -> Add    (融合 SSM + 门控, 一条路径搞定)

MambaLayer 内部:
    x -> Linear up (2D 分支: main + gate)
       main: ─ Conv1D ─ SiLU ─ SelectiveSSM ─
       gate: ─ SiLU   ─                       ⊙  ─ Linear down ─> out
    为什么需要 Conv1D?
        SelectiveSSM 是逐通道独立建模 (没有跨通道交互);
        1D depth-wise conv (kernel_size=4) 引入局部跨通道混合, 增强表达力,
        对应 Transformer 里 attention 的 "token 间交互"。
    为什么需要 gate 分支?
        借鉴 GLU/SwiGLU 门控思想, 让模型自动学"哪些通道的 SSM 输出该保留"。

本文件实现教学版 Mamba LM, 每层 = MambaLayer; 不再追加 FFN
(Mamba 论文实证 SSM + gate 已经足够, FFN 可省, 这也是 Mamba 参数效率高的原因)。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.sparse.ssm import SelectiveSSM


class MambaLayer(nn.Module):
    """
    单个 Mamba 层 (教学版 selective SSM + depth-wise conv + gate)

    数据流:
        x:          [B, T, D]
        x_and_gate: Linear_in(x) -> split → (x_main, x_gate) 各 [B, T, d_inner]
        x_main:     Conv1d(x_main) -> SiLU -> SelectiveSSM(x_main)
        y:          x_main ⊙ SiLU(x_gate)           (门控乘)
        out:        Linear_out(y)  [B, T, D]

    Args:
        d_model:   输入输出维度
        d_inner:   内部扩展维度, Mamba 默认 2*d_model
        d_state:   SelectiveSSM 的隐状态维度
        d_conv:    depth-wise conv kernel 大小
        dt_rank:   Δ 的低秩维度
    """

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()

        if d_inner is None:
            d_inner = 2 * d_model
        self.d_inner = d_inner
        self.d_conv = d_conv

        # 同时产出 main 与 gate 两分支, 减少一次 matmul
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # depth-wise 1D conv: groups=d_inner 保证每通道独立卷, 开销极小
        # padding=d_conv - 1 + 因果裁剪: 保持序列长度, 且只看过去的卷积窗口
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.ssm = SelectiveSSM(d_model=d_inner, d_state=d_state, dt_rank=dt_rank)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        B, T, _ = x.shape

        # 1) 一次投影出 main 与 gate
        x_and_gate = self.in_proj(x)                                  # [B, T, 2*d_inner]
        x_main, x_gate = x_and_gate.chunk(2, dim=-1)                  # each [B, T, d_inner]

        # 2) Conv1d 需要 [B, C, T]; 卷积后做因果裁剪 (去掉右侧 padding)
        x_main = x_main.transpose(1, 2)                               # [B, d_inner, T]
        x_main = self.conv1d(x_main)[:, :, :T]                        # 因果: 只保留前 T 步
        x_main = x_main.transpose(1, 2)                               # [B, T, d_inner]
        x_main = F.silu(x_main)

        # 3) Selective SSM 做时间建模
        x_main = self.ssm(x_main)                                     # [B, T, d_inner]

        # 4) gate 分支 + 门控乘
        y = x_main * F.silu(x_gate)

        # 5) 降回 d_model
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """
    标准 Mamba block: Pre-RMSNorm + MambaLayer + 残差

    数据流 (无 FFN, 无 attention):
        x -> RMSNorm -> MambaLayer -> Add
    """

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.layer = MambaLayer(
            d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(self.norm(x))


class Mamba(nn.Module):
    """
    Mamba 语言模型 (教学版)

    架构:
        idx -> TokenEmbed
            -> N x MambaBlock (Pre-RMSNorm + MambaLayer)
            -> RMSNorm
            -> lm_head (weight tied)

    无需因果 mask: SelectiveSSM 本身按时间 scan, 每步只读过去的状态,
    天然因果; 相比 Transformer 少了一个 O(T^2) 掩码大矩阵。

    Args:
        vocab_size: 词表大小
        d_model:    主干维度
        num_layers: Mamba 层数 (等参数量下通常比 Transformer 多 2-3×)
        d_state:    SSM 隐状态维度, 经验 16
        d_conv:     conv 窗口大小, 经验 4
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        d_inner: Optional[int] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Mamba 原实现: trunc_normal(std=0.02), 且 **不做** sqrt(d_model) 缩放
        # 避免 SSM scan 输入量级过大导致训练前期数值不稳
        nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)

        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model, d_inner=d_inner,
                    d_state=d_state, d_conv=d_conv,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: [B, T] token IDs
        Returns:
            logits: [B, T, vocab_size]
        """
        # Mamba 不乘 sqrt(d_model): 原论文做法, SSM 对输入量级敏感, 保持小 std 更稳
        x = self.token_embedding(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """朴素生成 (每步重算, 无 SSM 状态缓存); Mamba 真实部署会滚动缓存 h_t。

        教学注意: 随机初始化 + 纯 Python scan 可能让早期训练前的 logits 出现 nan,
        这里做 nan → 均匀分布的兜底, 让演示不至于崩。"""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # 数值安全: 任何 nan / inf 用 0 替换 (softmax 会把它们摊到均匀采样)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                # 若 softmax 仍全 0 (极端情形), 退化到 argmax
                if probs.sum().item() == 0:
                    idx_next = logits.argmax(dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
