#!/usr/bin/env python
"""
BERT 推理: 用断言验证 "双向" 和 "padding mask" 两件事
"""

import torch

from llm_models.models.language_models.bert import BERT


def main():
    torch.manual_seed(42)
    V, B, T = 1000, 2, 16
    model = BERT(vocab_size=V, d_model=128, n_heads=4, num_layers=2, max_len=64).eval()
    print(f"BERT Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    ids = torch.randint(1, V - 1, (B, T))
    with torch.inference_mode():
        logits, hidden = model(ids, return_hidden=True)
        assert logits.shape == (B, T, V) and hidden.shape == (B, T, 128)

        # 1) 双向: 改最后一个 token, 第 0 个位置的输出也变 (GPT 的因果 mask 下它不会变)
        ids2 = ids.clone(); ids2[:, -1] = (ids2[:, -1] + 1) % V
        d_first = (model(ids2)[:, 0] - logits[:, 0]).abs().max().item()
        print(f"改末尾 token → 位置 0 的 logits 变化 {d_first:.4f} (>0 即双向)")
        assert d_first > 1e-4

        # 2) padding mask: 后 4 个位置标成 pad 后, 无论 pad 处放什么 token, 有效位置的输出不变
        am = torch.ones(B, T, dtype=torch.long); am[:, -4:] = 0
        a = model(ids, attention_mask=am)[:, :-4]
        ids3 = ids.clone(); ids3[:, -4:] = torch.randint(1, V - 1, (B, 4))
        b = model(ids3, attention_mask=am)[:, :-4]
        d_pad = (a - b).abs().max().item()
        print(f"改 pad 处 token → 有效位置 logits 变化 {d_pad:.2e} (=0 即 mask 生效)")
        assert d_pad < 1e-5

        # 3) segment embedding 真的参与了计算
        tt = torch.zeros_like(ids); tt[:, T // 2:] = 1
        assert (model(ids, token_type_ids=tt) - logits).abs().max() > 1e-4


if __name__ == "__main__":
    main()
