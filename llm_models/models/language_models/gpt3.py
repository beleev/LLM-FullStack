"""
GPT-3 模型模块

Decoder-only Transformer (Brown et al., 2020, "Language Models are Few-Shot Learners")。

历史背景:
    GPT 系列把所有 NLP 任务统一为"文本续写", 训练目标只有一个 ——
    自回归地预测下一个 token。该范式的优势:
        - 无需任务特化结构, 一套权重适配一切
        - 海量纯文本即可预训练, 数据规模可任意扩大
        - 推理时通过 prompt 即可激发任务能力 (in-context learning)
    GPT-3 (175B 参数) 首次系统验证了"规模即能力"的 Scaling Law。

教学重点:
    - Pre-LN: x + dropout(sublayer(norm(x)))
      把 LayerNorm 放在残差分支内部 (而非 Post-LN 的"加完再 norm"),
      可显著稳定深层训练, 不再强依赖 learning-rate warmup。
    - 因果掩码: register_buffer 一次性缓存上三角 mask,
      避免每次 forward 都重建张量、节省显存与开销。
    - Weight Tying: lm_head.weight = token_embedding.weight
      共享输入/输出 embedding (Press & Wolf, 2017)。
      理由: 二者都是"token ↔ 向量空间"的映射, 共享可减半参数、
      并形成隐式正则化, 收敛更快。
    - generate(): 推理用 @torch.inference_mode() (比 no_grad() 更快,
      因跳过 autograd 版本计数)。这里走"每步重算全序列"的朴素实现,
      易读但低效; 工业部署需 KV cache 才能 O(1) 增量。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.core.position_encoding import (
    RotaryPositionalEncoding,
    SinPositionalEncoding,
)
from llm_models.utils.masks import build_causal_mask


def _make_gpt_block(
    d_model: int, n_heads: int, d_ff: int, dropout: float
) -> PreLNBlock:
    """按 GPT-3 的组件组合出 Pre-LN Block: MHA + GELU-FFN + LayerNorm。

    GELU 替代原始 ReLU: 平滑、可导、更适合大模型 (BERT/GPT-2 起的惯例)。
    """
    return PreLNBlock(
        d_model=d_model,
        attn=MultiHeadAttention(d_model, n_heads),
        ffn=GeLUFeedForward(d_model, d_ff),
        norm_cls=nn.LayerNorm,
        dropout=dropout,
    )


# 对外保留旧符号: GPTBlock 即带 GPT 组合的 PreLNBlock
GPTBlock = PreLNBlock


class GPT3(nn.Module):
    """
    GPT-3 Decoder-only LLM。

    架构:
        idx -> Embedding * sqrt(d_model) -> (Sin PE 或 RoPE)
            -> N x PreLNBlock(MHA + GELU-FFN + LN)
            -> LayerNorm -> lm_head (weight tied)

    设计要点:
        - Decoder-only: 自回归 LM, 一套结构覆盖理解+生成
        - d_ff = 4 * d_model: GPT 系列经验比例
        - 末尾 LayerNorm (ln_f): Pre-LN 结构必须的"出口规范化",
          否则残差累计会导致最后的输出范数过大
        - register_buffer("causal_mask"): 一次性构建上三角 mask 并随
          .to(device) 自动迁移; persistent=False 不进 state_dict
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 12,
        max_len: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_rope = use_rope
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if use_rope:
            # RoPE: 旋转位置编码, 作用于注意力头内部 Q/K, head_dim 即旋转维度
            d_head = d_model // n_heads
            self.pos_encoder = RotaryPositionalEncoding(d_head, max_len)
        else:
            # Sin-PE: 原始 Transformer 的绝对位置编码, 加在 token embedding 上
            self.pos_encoder = SinPositionalEncoding(d_model, max_len)

        d_ff = 4 * d_model  # GPT-3 使用 4x 扩展, 经验最优容量比
        self.layers = nn.ModuleList(
            [_make_gpt_block(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        # bias=False: lm_head 与 embedding 共享权重, embedding 无 bias
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight Tying: 将输出投影矩阵与输入 embedding 共用同一份权重
        # 参数量减半 + 隐式正则化, 是 GPT-2 起的常规做法
        self.lm_head.weight = self.token_embedding.weight

        # 预构建 max_len × max_len 因果 mask, 避免每次 forward 重建
        causal = build_causal_mask(max_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        """按当前序列长度切片缓存 mask; 超长时即兴构建 (容错路径)。"""
        if seq_len <= self.causal_mask.size(-1):
            return self.causal_mask[:, :seq_len, :seq_len]
        return build_causal_mask(seq_len, self.causal_mask.device)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: [B, T] token ids
        Returns:
            logits: [B, T, vocab_size]
        """
        _, seq_len = idx.size()
        if seq_len > self.max_len:
            raise ValueError(
                f"序列长度 {seq_len} 超过最大上下文窗口 {self.max_len}"
            )

        # emb * sqrt(d_model): 让 embedding 与 PE/残差主干量级匹配
        x = self.token_embedding(idx) * math.sqrt(self.d_model)

        # 位置编码两条路径二选一:
        # - RoPE: 不改主干张量, 传给每层 attention 对 Q/K 旋转
        # - Sin-PE: 直接加到 embedding 上, 后续层不再感知位置
        rope_handler: Optional[nn.Module] = None
        if self.use_rope:
            rope_handler = self.pos_encoder
        else:
            x = self.pos_encoder(x)

        mask = self._causal_mask(seq_len)
        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope_handler)

        # Pre-LN 架构末端的统一规范化
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成 (朴素实现: 每步重算全序列)

        教学取舍:
            真实部署会维护 K/V cache, 每步只算新 token 的 attention,
            复杂度从 O(T^2) 降到 O(T); 这里为了让训练/推理共用同一条
            forward, 牺牲性能换可读性。
            @torch.inference_mode() 代替 torch.no_grad():
            额外跳过 autograd 版本计数, 速度略快。

        Args:
            idx:            [B, T] 起始 prompt ids
            max_new_tokens: 续写 token 数
            temperature:    softmax 温度, <1 锐化, >1 平滑
            top_k:          仅在概率最高的 k 个候选中采样 (截断采样)
        Returns:
            [B, T + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 上下文超长时只保留最近 max_len 个 token (滑动窗口截断)
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]

            logits = self(idx_cond)
            # 只取最后一个位置的 logits 用来预测下一个 token
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # 把第 k 名以下的 logits 置为 -inf, softmax 后概率为 0
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            # 多项式采样 (而非 argmax) 引入随机性, 提升多样性
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
