"""
MTP — Multi-Token Prediction (多 token 预测) 模块

论文出处:
    "Better & Faster Large Language Models via Multi-token Prediction"
        (Gloeckle et al., Meta, 2024 — 并行多 head 版)
    DeepSeek-V3 Technical Report (2024) — 串行因果链版 (本实现)

动机:
    标准 LM 的每个位置只预测 t+1 一个 token。MTP 让位置 t 同时预测
    t+1, t+2, ..., t+1+K, 带来三重收益:
      1. **训练信号更密** — 同一条数据提供 (K+1) 份监督, 数据效率更高
      2. **表征被迫"向前规划"** — hidden state 必须编码更远期的信息,
         缓解 next-token 短视 (teacher forcing 只看一步)
      3. **推理免费拿草稿** — MTP head 对 t+2 的预测可直接作为投机解码
         (speculative decoding) 的 draft; DeepSeek-V3 报告草稿接受率 85%+,
         解码加速约 1.8x (对应 llm_infer/m07_speculative_decoding)

DeepSeek-V3 的串行式 MTP (与 Gloeckle 的并行独立 head 不同, 保留完整因果链):

    主干:    h_i^0 = Backbone(t_<=i)                       → head → 预测 t_{i+1}
    MTP-1:   h_i^1 = Block_1( W_1 [RMSNorm(h_i^0) ; RMSNorm(Emb(t_{i+1}))] )
                                                           → head → 预测 t_{i+2}
    MTP-k:   同构地堆叠, 每深一级多看一个真实 token (teacher forcing)

    要点: Embedding 与 lm_head 与主干**共享**, 每个 MTP 模块只新增
          一个拼接投影 W_k 和一个 Transformer Block, 参数开销极小。

损失:
    L = L_main + λ · mean_k( L_mtp_k )       DeepSeek-V3: λ = 0.3 (前期) / 0.1

推理:
    部署时可以直接丢弃 MTP 模块 (零成本), 或保留用作投机解码 draft head。
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import GroupedQueryAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import SwiGLUFeedForward
from llm_models.layers.core.normalization import RMSNorm
from llm_models.layers.core.position_encoding import RotaryPositionalEncoding
from llm_models.training.loss import LossComputer
from llm_models.utils.masks import build_causal_mask, combine_causal_and_padding_mask


class MTPModule(nn.Module):
    """
    单个 MTP 级联模块 (DeepSeek-V3 式)。

    输入上一级的 hidden h^{k-1} 与"下一个真实 token"的 embedding, 输出本级
    hidden h^k (用共享 lm_head 解码即得 t_{i+1+k} 的预测)。

        h^k = Block( W [RMSNorm(h^{k-1}) ; RMSNorm(emb_next)] )

    为什么拼接后要先各自 RMSNorm:
        h 与 embedding 的数值尺度不同 (h 经过了多层残差累加), 直接拼接会让
        投影矩阵 W 先花容量学"对齐尺度"; 各自归一化后拼接更稳。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_kv_heads: Optional[int],
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_hidden = RMSNorm(d_model)
        self.norm_emb = RMSNorm(d_model)
        # 拼接 [h; emb] (2d) 投影回 d, 是 MTP 模块唯一的"新零件"
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = PreLNBlock(
            d_model=d_model,
            attn=GroupedQueryAttention(
                d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
            ),
            ffn=SwiGLUFeedForward(d_model, d_ff),
            norm_cls=RMSNorm,
            dropout=dropout,
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        h_prev: torch.Tensor,        # [B, T, D] 上一级 hidden
        emb_next: torch.Tensor,      # [B, T, D] 真实 next-token 的 embedding
        mask: torch.Tensor,
        rope: RotaryPositionalEncoding,
    ) -> torch.Tensor:
        x = torch.cat([self.norm_hidden(h_prev), self.norm_emb(emb_next)], dim=-1)
        x = self.proj(x)                      # [B, T, 2D] -> [B, T, D]
        return self.block(x, mask=mask, rope=rope)


class MTPLLaMA(nn.Module):
    """
    LLaMA 主干 + K 级串行 MTP 模块 (教学版)。

    forward 返回 dict:
        {
          "logits":     [B, T, V]          主 head (预测 t+1)
          "mtp_logits": List[[B, T, V]]    第 k 项预测 t+1+k
        }

    Args:
        vocab_size / d_model / n_heads / num_kv_heads / num_layers / max_len /
        d_ff / dropout: 同 LLaMA。
        mtp_depth: MTP 级联深度 K (DeepSeek-V3 取 1)。
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        num_layers: int = 32,
        max_len: int = 4096,
        mtp_depth: int = 1,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mtp_depth < 1:
            raise ValueError(f"mtp_depth 至少为 1, 当前 {mtp_depth}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.mtp_depth = mtp_depth

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        d_head = d_model // n_heads
        self.rope = RotaryPositionalEncoding(d_head, max_len)

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        # ---- 主干: 与 LLaMA 相同的 N 层 stack ----
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=GroupedQueryAttention(
                        d_model=d_model, num_heads=n_heads, num_kv_heads=num_kv_heads,
                    ),
                    ffn=SwiGLUFeedForward(d_model, d_ff),
                    norm_cls=RMSNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)

        # ---- 共享输出头: 主干与所有 MTP 模块共用 (也与 embedding 绑定) ----
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # ---- K 个串行 MTP 模块 ----
        self.mtp_modules = nn.ModuleList(
            [
                MTPModule(d_model, n_heads, num_kv_heads, d_ff, dropout)
                for _ in range(mtp_depth)
            ]
        )

        causal = build_causal_mask(max_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(idx) * math.sqrt(self.d_model)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            idx: [B, T] token IDs
        Returns:
            {"logits": [B, T, V], "mtp_logits": List of [B, T, V]}
        """
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")

        causal = self._causal_mask(T)
        mask = combine_causal_and_padding_mask(causal, attention_mask)

        # ---- 主干前向 ----
        h = self._embed(idx)
        for layer in self.layers:
            h = layer(h, mask=mask, rope=self.rope)
        logits_main = self.lm_head(self.ln_f(h))

        # ---- MTP 级联: 第 k 级在位置 i 处拼接真实 token t_{i+k} 的 embedding ----
        mtp_logits: List[torch.Tensor] = []
        for k, module in enumerate(self.mtp_modules, start=1):
            # teacher forcing: idx 左移 k 位; 末尾 k 个位置没有未来 token,
            # 用 0 占位 (这些位置的预测会在 MTPLoss 里被 -100 屏蔽)
            shifted = torch.zeros_like(idx)
            shifted[:, :-k] = idx[:, k:]
            emb_next = self._embed(shifted)

            h = module(h, emb_next, mask=mask, rope=self.rope)
            mtp_logits.append(self.lm_head(module.final_norm(h)))

        return {"logits": logits_main, "mtp_logits": mtp_logits}


class MTPLoss(LossComputer):
    """
    MTP 联合损失:  L = CE(main) + λ · mean_k CE(mtp_k)

    标签对齐 (labels[i] = t_{i+1} 是标准 next-token 标签):
        MTP-k 在位置 i 预测 t_{i+1+k} = labels[i+k]
        → 把 labels 左移 k 位作为第 k 级的目标, 末尾 k 个位置置 -100

    Args:
        mtp_lambda: MTP 分支权重 λ (DeepSeek-V3: 0.3 → 0.1)
        ignore_index: 同 cross_entropy 约定
    """

    def __init__(self, mtp_lambda: float = 0.3, ignore_index: int = -100) -> None:
        self.mtp_lambda = mtp_lambda
        self.ignore_index = ignore_index

    def _ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

    def compute(
        self,
        model_output: Dict[str, Any],
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        main_loss = self._ce(model_output["logits"], labels)

        mtp_losses: List[torch.Tensor] = []
        for k, logits_k in enumerate(model_output["mtp_logits"], start=1):
            labels_k = torch.full_like(labels, self.ignore_index)
            labels_k[:, :-k] = labels[:, k:]   # 目标整体左移 k 位
            mtp_losses.append(self._ce(logits_k, labels_k))

        mtp_loss = torch.stack(mtp_losses).mean()
        total = main_loss + self.mtp_lambda * mtp_loss
        return {
            "total_loss": total,
            "main_loss": main_loss.detach(),
            "mtp_loss": mtp_loss.detach(),
        }
