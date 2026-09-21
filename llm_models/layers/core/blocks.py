"""
通用 Transformer Block — Pre-LN 模板 + 依赖注入

是什么: PreLNBlock (self-attn + FFN) 与 PreLNCrossBlock (self-attn + cross-attn + FFN)。
解决什么: Post-LN `x = LN(x + f(x))` 的残差主路上有 LN, 深层梯度不稳、必须 warmup;
          Pre-LN `x = x + f(LN(x))` (Xiong et al., 2020) 主路是纯恒等, 梯度直通。
怎么用: attn / ffn / norm_cls 由调用方注入 —— 模型间的差异全在构造参数里:
    GPT-3 = MHA + GELU-FFN + LayerNorm;  LLaMA = GQA + SwiGLU + RMSNorm;  DeepSeek-V3 = MLA + MoE + RMSNorm
读代码时盯住: `residual` —— 它从不经过 norm。
"""

from typing import Callable, Optional

import torch
import torch.nn as nn


class PreLNBlock(nn.Module):
    """
    Pre-LN 自注意力 + FFN Block (Decoder-only LLM 标准模块)

    数据流 (注意 LN 在残差分支内、不在主路径上):
        x ──┬── norm1 ── attn(self) ─┐
            │                         ⊕ ──┬── norm2 ── ffn ─┐
            └──────────────────────── ─┘   │                  ⊕ ── out
                                           └───────────────── ─┘

    Args:
        d_model: 模型维度
        attn: 自注意力模块。必须实现 forward(q, k, v, mask, rope, position_ids[, cache])
              并返回张量 (不返回 tuple)
        ffn: 前馈模块。必须实现 forward(x) -> x
        norm_cls: 归一化工厂函数 (默认 LayerNorm；现代 LLM 通常传 RMSNorm)
        dropout: 残差分支 dropout 概率 (训练用，预训练大模型一般设很小或 0)
    """

    def __init__(
        self,
        d_model: int,
        attn: nn.Module,
        ffn: nn.Module,
        norm_cls: Callable[[int], nn.Module] = nn.LayerNorm,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        # 子层 1: 自注意力分支 (Pre-LN: 归一化只作用于进入 attn 的拷贝)
        residual = x
        h = self.norm1(x)
        # cache 只在给定时才下传: 不支持 KV cache 的 attn 模块 (教学版 MHA 等) 无需改签名
        extra = {} if cache is None else {"cache": cache}
        h = self.attn(q=h, k=h, v=h, mask=mask, rope=rope, position_ids=position_ids, **extra)
        x = residual + self.dropout(h)

        # 子层 2: FFN 分支
        residual = x
        h = self.norm2(x)
        h = self.ffn(h)
        x = residual + self.dropout(h)
        return x


class PreLNCrossBlock(nn.Module):
    """
    Pre-LN Decoder Block: Masked Self-Attn + Cross-Attn + FFN

    用于经典 encoder-decoder (翻译、Seq2Seq、T5) 或多模态视觉编码器到语言解码器的桥接。
    self_attn 看 decoder 内部因果序列，cross_attn 让 decoder 去 "查询" encoder 输出。

    数据流:
        x -> norm1 -> self_attn (masked)              -> Add
          -> norm2 -> cross_attn (Q=x, K/V=context)   -> Add
          -> norm3 -> ffn                             -> Add

    Args:
        d_model: 模型维度
        self_attn: 自注意力模块 (带因果掩码)
        cross_attn: 交叉注意力模块 (Q 来自 decoder，K/V 来自 encoder context)
        ffn: FFN 模块
        norm_cls: 归一化工厂 (默认 LayerNorm)
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        self_attn: nn.Module,
        cross_attn: nn.Module,
        ffn: nn.Module,
        norm_cls: Callable[[int], nn.Module] = nn.LayerNorm,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.norm3 = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 子层 1: 因果自注意力 (decoder 内部时序)
        residual = x
        h = self.norm1(x)
        h = self.self_attn(q=h, k=h, v=h, mask=self_mask, rope=rope, position_ids=position_ids)
        x = residual + self.dropout(h)

        # 子层 2: Cross-Attention (Q 来自 decoder，K/V 来自 encoder)
        # RoPE 显式置 None：Q 和 K 处于不同的位置空间 (decoder 时序 vs encoder 序列)，
        # 强行加同一套相对位置旋转会引入错误信号
        residual = x
        h = self.norm2(x)
        h = self.cross_attn(q=h, k=context, v=context, mask=context_mask, rope=None)
        x = residual + self.dropout(h)

        # 子层 3: FFN
        residual = x
        h = self.norm3(x)
        h = self.ffn(h)
        x = residual + self.dropout(h)
        return x
