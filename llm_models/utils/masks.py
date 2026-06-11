"""
掩码（Mask）工具模块

注意力机制中常用的两类掩码：
1. Padding Mask（填充掩码）：屏蔽 batch 内填充位置（pad token），避免模型把 pad
   当成有效 token 参与注意力计算 —— 否则不仅浪费算力，还会让真实 token 的表示
   被无意义的 pad 信息污染（信息泄露）。
2. Causal Mask / Subsequent Mask（因果掩码 / 下三角掩码）：用于自回归语言模型
   （如 Decoder、GPT）。每个位置只能"看到"自己及之前的 token，不能看未来的
   token —— 否则训练时模型可以直接抄答案（next token），丧失泛化能力。

实现细节：
- 本模块的掩码使用 bool 语义：True = 保留 / 可见，False = 屏蔽。
- 在注意力打分（QK^T）阶段，会把 False 位置的分数加上 -inf；
  这样 softmax 后该位置权重 ≈ 0，从而真正"看不到"被屏蔽位置。
- 用 -inf 而不是直接置 0 的原因：softmax 是对所有位置归一化的，
  必须先把屏蔽位置压到 -inf，归一化后才会得到 0。

提供函数：
- get_pad_mask: 由 token id 序列生成 padding mask
- get_subsequent_mask: 由 token id 序列生成因果掩码
- build_causal_mask: 因果掩码的通用版本（接受 seq_len + device，
  当只有 embedding 没有 token id 时使用）
- build_sliding_window_mask: 带状因果掩码（滑动窗口注意力 SWA，Mistral / Gemma /
  GPT-OSS），可选保留开头 sink token（StreamingLLM / GPT-OSS attention sink）
- combine_causal_and_padding_mask: 合并因果掩码与 padding mask
- combine_masks: 通用的两个掩码 AND 合并
"""

import torch
from typing import Optional


def get_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成 Padding 掩码

    背景：训练时为了 batch 化，必须把不同长度的句子用 pad token 补齐。
    pad 本身没有任何语义，必须在注意力中屏蔽掉，否则：
    - pad 会和真实 token 互相影响，污染表示；
    - softmax 分母里多出 pad 项，权重分布失真。

    Args:
        seq: 输入序列张量 [batch, seq_len]，元素为 token id
        pad_idx: 填充 token 的索引，默认为 0

    Returns:
        掩码张量 [batch, 1, seq_len]
        - True 表示有效位置（保留）
        - False 表示填充位置（屏蔽）

        中间多出的维度 1 是为了方便后续与因果掩码 [1, T, T] / 注意力分数
        [B, H, T, T] 做广播 AND 运算。

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        >>> mask = get_pad_mask(seq, pad_idx=0)
        >>> # mask: [[True, True, True, False, False], [True, True, False, False, False]]
    """
    # 逐元素比较 → bool 张量；unsqueeze(1) 把形状由 [B, T] 变为 [B, 1, T]
    # 多出的中间维度用于和 [1, T, T] 因果掩码 / [B, H, T, T] 注意力分数广播
    return (seq != pad_idx).unsqueeze(1)


def get_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    生成因果掩码（Subsequent Mask / Look-ahead Mask）

    用于 Decoder / GPT 类自回归模型的自注意力：
    每个位置 t 只能看到位置 0..t（包含自己），不能看到 t+1..T-1（未来）。
    若不加这个掩码，训练时模型在预测位置 t 的下一个 token 时，可以直接
    "偷看"目标 token 本身，等于训练阶段就给了答案，模型学不到任何东西。

    Args:
        seq: 输入序列张量 [batch, seq_len]（这里只用其 seq_len 与 device）

    Returns:
        下三角掩码张量 [1, seq_len, seq_len]
        - True 表示可以看到的位置（下三角 + 对角线）
        - False 表示需要遮蔽的位置（上三角，未来 token）

        最前面的维度 1 用于和 batch 维度广播，整个 batch 共用同一份因果掩码。

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 4]])
        >>> mask = get_subsequent_mask(seq)
        >>> # mask 形状: [1, 4, 4]
        >>> # [[True,  False, False, False],   # 位置 0 只能看 0
        >>> #  [True,  True,  False, False],   # 位置 1 能看 0,1
        >>> #  [True,  True,  True,  False],   # 位置 2 能看 0,1,2
        >>> #  [True,  True,  True,  True ]]   # 位置 3 能看 0,1,2,3
    """
    sz = seq.size(1)
    return build_causal_mask(sz, seq.device)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    生成因果掩码（通用版本）

    与 get_subsequent_mask 功能相同，但接受 seq_len + device 而非 seq Tensor，
    适合在已经拥有 embedding 但没有原始 token IDs 时调用（例如多模态模型，
    输入是图像/音频特征拼接而成的 hidden states）。

    Args:
        seq_len: 序列长度
        device: 目标设备（与输入张量保持一致，避免 CPU/GPU 之间隐式拷贝）

    Returns:
        下三角掩码张量 [1, seq_len, seq_len]
        - True 表示可以看到的位置
        - False 表示需要遮蔽的位置
    """
    # torch.triu(..., diagonal=1) 取严格上三角（不含对角线）→ 这里就是"未来位置"
    # 然后 == 0 反转，得到"非未来位置 = 可见"的下三角 bool 矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return (mask == 0).unsqueeze(0)


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    sink_tokens: int = 0,
) -> torch.Tensor:
    """
    生成滑动窗口（带状）因果掩码 — Sliding Window Attention (SWA)

    全因果掩码下，位置 t 能看到 [0, t] 共 t+1 个位置，注意力计算与 KV cache
    都随序列长度线性/平方增长。SWA 把可见范围截断成最近 W 个位置：

        可见(t, s) = (s <= t) 且 (s > t - W)

    于是注意力矩阵从"下三角"变成"带状下三角"：
      - 单层感受野被限制在 W 内，但信息可以跨层接力 —— L 层的理论感受野 ≈ L·W
        (Mistral-7B: 32 层 × 4096 窗口 ≈ 131K)
      - 推理时 KV cache 只需保留最近 W 个位置（rolling buffer 环形覆写），
        显存从 O(T) 封顶到 O(W)

    sink_tokens > 0 时额外保留开头 S 个位置永远可见。这是 StreamingLLM (2023)
    的发现：softmax 必须把注意力分给"某些位置"，模型训练后习惯把多余注意力
    倾倒在开头几个 token 上（attention sink）。如果窗口滑过把它们逐出 cache，
    输出分布会崩坏；保留 4 个 sink 即可在无限流式输入下保持质量。
    GPT-OSS (2025) 进一步把 sink 做成了每个 head 可学习的 logit。

    Args:
        seq_len: 序列长度
        window_size: 窗口大小 W（>= 1）；W >= seq_len 时退化为全因果掩码
        device: 目标设备
        sink_tokens: 额外永远可见的开头位置数 S（默认 0 = 纯 SWA）

    Returns:
        带状掩码 [1, seq_len, seq_len]，True = 可见，False = 屏蔽

    Example:
        >>> build_sliding_window_mask(5, window_size=2, device="cpu")[0].int()
        tensor([[1, 0, 0, 0, 0],     # 位置 0 只看自己
                [1, 1, 0, 0, 0],     # 位置 1 看 {0, 1}
                [0, 1, 1, 0, 0],     # 位置 2 看 {1, 2} — 0 滑出窗口
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1]])
    """
    if window_size < 1:
        raise ValueError(f"window_size 至少为 1, 当前 {window_size}")

    i = torch.arange(seq_len, device=device).unsqueeze(1)  # query 位置 [T, 1]
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # key 位置   [1, T]

    # 带状下三角: 因果 (j <= i) 且 在窗口内 (j > i - W)
    visible = (j <= i) & (j > i - window_size)

    if sink_tokens > 0:
        # 开头 S 个位置对所有"未来"位置永远可见 (仍需满足因果性 j <= i)
        visible = visible | ((j < sink_tokens) & (j <= i))

    return visible.unsqueeze(0)


def combine_causal_and_padding_mask(
    causal_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    将因果掩码与 padding 掩码合并。

    使用场景：自回归模型（GPT/Decoder）训练时，既要防止偷看未来 token
    （因果掩码），又要忽略 batch 内的 pad 位置（padding 掩码）。
    合并方式是按位 AND：两个掩码都为 True 时才算可见。

    Args:
        causal_mask: [1, T, T]，下三角因果掩码
        attention_mask: [B, T]，True 表示有效 token；可为 None
            （None 表示 batch 内没有 pad，例如推理或定长输入）

    Returns:
        combined_mask: [B, T, T] 或 [1, T, T]（当 attention_mask 为 None 时）
    """
    if attention_mask is None:
        return causal_mask

    attention_mask = attention_mask.bool()
    # 注意：只屏蔽 key（被关注侧）即可。query 侧若是 pad，对应位置的输出
    # 本来就会在 loss 里被忽略（label=-100），不必再额外屏蔽。
    # key_mask: [B, T] -> [B, 1, T] -> [B, T, T]
    key_mask = attention_mask.unsqueeze(1).expand(-1, causal_mask.size(1), -1)
    return key_mask & causal_mask


def combine_masks(pad_mask: torch.Tensor, subsequent_mask: torch.Tensor) -> torch.Tensor:
    """
    组合 Padding 掩码和因果掩码

    用于 Decoder：需要同时遮蔽填充位置和未来位置。
    通过广播 + 按位 AND 合并：
      pad_mask:        [B, 1, T]   ─┐
                                    ├─ broadcast → [B, T, T]
      subsequent_mask: [1, T, T]   ─┘

    Args:
        pad_mask: Padding 掩码 [batch, 1, seq_len]
        subsequent_mask: 因果掩码 [1, seq_len, seq_len]

    Returns:
        组合掩码 [batch, seq_len, seq_len]，True = 可见，False = 屏蔽

    Example:
        >>> tgt_pad_mask = get_pad_mask(tgt, pad_idx=0)
        >>> tgt_subsequent_mask = get_subsequent_mask(tgt)
        >>> tgt_mask = combine_masks(tgt_pad_mask, tgt_subsequent_mask)
    """
    return pad_mask & subsequent_mask
