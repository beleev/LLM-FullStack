#!/usr/bin/env python
"""
BERT (Encoder-only) 前向示例

展示:
- 双向注意力 (无因果 mask) 的编码
- 每个位置都能看到整句, 因此输出可直接用于 token 级分类任务
- MLM head 输出 [B, T, vocab] logits, 与 GPT 同形但语义不同
"""

import torch
from llm_models.models.language_models.bert import BERT


def main():
    torch.manual_seed(42)

    vocab_size = 1000
    batch, seq_len = 2, 16

    model = BERT(
        vocab_size=vocab_size, d_model=128, n_heads=4,
        num_layers=2, max_len=64,
    ).eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"BERT Mini | 参数量: {num_params:,}")

    input_ids = torch.randint(1, vocab_size - 1, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attention_mask)

    print(f"输入:  {tuple(input_ids.shape)}")
    print(f"输出:  {tuple(logits.shape)}  (MLM head, 与 GPT 同形但是双向编码的结果)")
    assert logits.shape == (batch, seq_len, vocab_size)
    print("✅ BERT 前向通过")


if __name__ == "__main__":
    main()
