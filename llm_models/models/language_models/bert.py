"""
BERT — Encoder-only 双向 Transformer + 掩码语言建模 (Devlin et al., 2019)

是什么: 和 GPT 同一套 block, 只改两处 —— 不加因果 mask (每个位置看全句); 训练目标从"预测下一个"换成"还原被遮住的 token"。
解决了什么: GPT 的单向注意力让每个 token 只看得到左边, 做理解类任务 (分类/抽取/检索) 吃亏。
关键数字: 选 15% 位置 → 其中 80% 换 [MASK] / 10% 换随机 token / 10% 不变; loss 只在这 15% 上算 (其余 label = -100)。
         位置编码是可学习绝对位置; 另有 segment embedding 区分句子 A/B。
读代码时盯住: mask —— 这里只有 padding mask [B, 1, T], 没有下三角。对照 GPT 的 causal mask 看。
"""

from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.utils.init import init_weights


class BERTEmbeddings(nn.Module):
    """token + position (可学习) + segment 三者相加, 再 LayerNorm + Dropout。"""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, type_vocab_size: int = 2, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = input_ids.size(1)
        position_ids = torch.arange(T, device=input_ids.device)        # [T], 广播到 batch
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )                                                              # [B, T, D]
        return self.dropout(self.ln(x))


class BERT(nn.Module):
    """
    input_ids -> BERTEmbeddings -> N × PreLNBlock(MHA + GELU-FFN, 无因果 mask) -> LN -> MLM head

    原论文是 Post-LN; 这里沿用全库统一的 Pre-LN。[CLS]/[SEP] 由 tokenizer 构造, 模型不特殊处理。
    BERT-base: d_model=768, n_heads=12, num_layers=12, max_len=512。
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
        self.d_model = d_model
        self.max_len = max_len

        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, type_vocab_size, dropout)
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff or 4 * d_model),
                    norm_cls=nn.LayerNorm,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # MLM head: 先过一层 "变换", 再用与 token embedding 共享的矩阵投回词表
        self.mlm_transform = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.mlm_head = nn.Linear(d_model, vocab_size, bias=True)
        self.mlm_head.weight = self.embeddings.token_embeddings.weight

        # 默认 N(0,1) embedding + 权重共享 ⇒ 初始 logits std≈sqrt(D), 初始 CE ≈ 40 而不是 ln V
        init_weights(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        input_ids [B, T]; attention_mask [B, T] (1=有效, 0=pad); token_type_ids [B, T] ∈ {0,1}
        -> logits [B, T, V] (return_hidden=True 时再返回 hidden [B, T, D], 下游任务取 hidden[:, 0] 即 [CLS])
        """
        T = input_ids.size(1)
        if T > self.max_len:
            raise ValueError(f"序列长度 {T} 超过 max_len={self.max_len}")

        x = self.embeddings(input_ids, token_type_ids)                 # [B, T, D]
        # 只屏蔽 pad 列; [B, 1, T] 广播到 [B, T, T]。没有下三角 ⇒ 双向
        mask = None if attention_mask is None else attention_mask.bool().unsqueeze(1)

        for layer in self.layers:
            x = layer(x, mask=mask)

        hidden = self.ln_f(x)                                          # [B, T, D]
        logits = self.mlm_head(self.mlm_transform(hidden))             # [B, T, V]
        return (logits, hidden) if return_hidden else logits
