"""
BERT 模型模块

论文出处:
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    (Devlin et al., NAACL 2019)

Transformer 演进的另一条主线 (与 GPT 的 decoder-only 对立):
    Encoder-only, 双向自注意力, 掩码语言建模 (MLM) 预训练。
    - 每个位置能看到 **前后** 所有 token, 是理解类任务的天然选择
    - 预训练用 MLM: 随机 mask 输入 15% 的 token, 让模型重建它们
    - 下游任务只需加一个分类头就能微调

教学重点:
    - 与 GPT3 对比: 同样的 PreLN + MHA 骨架, 只是
        * attention mask: GPT 用因果下三角, BERT 用全可见 (padding mask 即可)
        * 训练目标: GPT 预测下一个 token, BERT 重建被 mask 的 token
        * 位置编码: 原论文用可学习绝对位置 (learnable), 本实现保留这种选择
    - Segment embedding: 句子对任务 (NSP) 需要区分"句子 A vs B",
        本教学实现保留 token_type_ids 接口 (简化版 NSP 未实现)
    - [CLS] / [SEP]: token 本身由外部 tokenizer 构造, 模型侧不做特殊处理

现代继承者:
    RoBERTa (去掉 NSP + 动态 mask), ALBERT (参数共享), DeBERTa (解耦 PE),
    E5 / BGE (句子嵌入模型) 等仍以 BERT 骨架为核心。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward


class BERTEmbeddings(nn.Module):
    """
    BERT 的三合一嵌入: token + position + segment

    - token:    词表嵌入
    - position: 可学习绝对位置嵌入 (原论文做法; 现代模型更爱 RoPE 但 BERT 保持原样)
    - segment:  token_type_id ∈ {0, 1}, 区分句子 A / 句子 B (NSP 任务用)

    三者相加后接 LayerNorm + Dropout (BERT 的"输入归一化"习惯)。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)

        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.ln(x))


class BERT(nn.Module):
    """
    BERT-base 教学版 (Encoder-only, 双向)

    架构:
        input_ids -> BERTEmbeddings (tok + pos + seg)
                  -> N x PreLNBlock(MHA + GELU-FFN + LayerNorm)  [无因果 mask]
                  -> LayerNorm
                  -> MLM head (权重与 token embedding 共享)

    与 GPT3 的关键差异:
        - 注意力 mask: 仅 padding mask (没有因果下三角), 实现双向注意力
        - 输出用途:   每个位置的 hidden 都可用 (下游 token classification);
                      [CLS] 位置通常作为 "pooler" 用于句子级任务
        - 训练目标:   MLM (外部提供被 mask 的 labels, 其他位置填 -100)

    Args:
        vocab_size:      词表大小
        d_model:         模型维度 (BERT-base 768, BERT-large 1024)
        n_heads:         注意力头数 (BERT-base 12, BERT-large 16)
        num_layers:      Transformer 层数 (BERT-base 12, BERT-large 24)
        max_len:         位置嵌入最大长度 (BERT 默认 512)
        type_vocab_size: segment 种类数, 默认 2
        d_ff:            FFN 隐藏维度, 默认 4 * d_model
        dropout:         Dropout 概率
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 12,
        max_len: int = 512,
        type_vocab_size: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.max_len = max_len

        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
        )

        # Encoder 层: MHA + GELU-FFN + LayerNorm, 无因果掩码
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                    norm_cls=nn.LayerNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # MLM head: 一个 "变换" + 输出投影; 输出投影权重与 token embedding 共享
        self.mlm_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.mlm_head = nn.Linear(d_model, vocab_size, bias=True)
        # Weight tying (BERT 原实现): MLM 输出矩阵共享 token embedding
        self.mlm_head.weight = self.embeddings.token_embeddings.weight

    @staticmethod
    def _padding_mask(attention_mask: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
        """
        把 [B, T] 的 padding mask 扩成 [B, 1, T] 供 attention 广播。

        为什么只要 padding mask 不要因果 mask?
            BERT 是双向理解模型; 每个位置都要能看到全句, 因此不需要下三角。
            pad 位置仍要屏蔽, 避免 attention 把 pad 当成有效 token。
        """
        if attention_mask is None:
            return None
        return attention_mask.bool().unsqueeze(1)  # [B, 1, T]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        Args:
            input_ids:      [B, T] token IDs
            attention_mask: [B, T] 1=有效, 0=pad; None 表示全部有效
            token_type_ids: [B, T] segment IDs (0/1); None 视为全 0
            return_hidden:  若为 True 则同时返回最后一层 hidden
        Returns:
            logits: [B, T, vocab_size] — MLM 头输出
            (可选) hidden: [B, T, d_model]
        """
        _, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")

        x = self.embeddings(input_ids, token_type_ids)
        mask = self._padding_mask(attention_mask, T)

        for layer in self.layers:
            x = layer(x, mask=mask)

        hidden = self.ln_f(x)
        logits = self.mlm_head(self.mlm_transform(hidden))

        if return_hidden:
            return logits, hidden
        return logits
