#!/usr/bin/env python
"""
Transformer 模型示例

演示标准 Encoder-Decoder Transformer 的使用方法（即原版 "Attention Is
All You Need" 架构，常用于机器翻译类 seq2seq 任务）。

关键点：
- src 经 Encoder 编码出 memory；tgt 在 Decoder 中以自回归方式生成
- src_mask（padding 掩码）：屏蔽源端的 pad token
- tgt_mask = padding 掩码 & 因果掩码：既屏蔽目标端 pad，又防止 Decoder
  在训练时偷看未来 token（teacher forcing 必备）
- Decoder 内还会用 cross-attention 让 tgt 关注 src 的 memory
"""

import torch
from llm_models.models import Transformer
from llm_models.utils import get_pad_mask, get_subsequent_mask


def main():
    torch.manual_seed(42)

    # --- 参数配置 ---
    # 源语言与目标语言词表可以不同（例如英→中翻译）
    src_vocab_size = 100
    tgt_vocab_size = 120

    d_model = 64
    n_heads = 4
    num_layers = 3
    d_ff = 128         # FFN 中间层维度，通常为 d_model 的 2-4 倍

    batch_size = 1
    src_len = 5
    tgt_len = 6

    print("=" * 50)
    print("Transformer 模型测试")
    print("=" * 50)
    print(f"源词表大小: {src_vocab_size}")
    print(f"目标词表大小: {tgt_vocab_size}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {n_heads}")
    print(f"层数: {num_layers}")

    # --- 实例化 Transformer ---
    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, n_heads, num_layers, d_ff,
        use_rope=True
    )
    model.eval()  # Demo 只做推理：关闭 dropout
    print("\n✓ Transformer 模型构建成功！")

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # --- 模拟输入数据 ---
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # --- 生成 Masks ---
    # src 端：只需 padding mask（Encoder 是双向注意力，可看见整段）
    src_mask = get_pad_mask(src, pad_idx=0)
    # tgt 端：padding mask + 因果 mask 必须组合
    # - padding mask: 屏蔽 pad 位置
    # - 因果 mask:    屏蔽未来 token（防止训练时偷看答案）
    tgt_pad_mask = get_pad_mask(tgt, pad_idx=0)
    tgt_lookahead_mask = get_subsequent_mask(tgt)
    tgt_mask = tgt_pad_mask & tgt_lookahead_mask  # 广播 AND: [B,1,T] & [1,T,T] → [B,T,T]

    print(f"\n输入 Src 形状: {src.shape}")
    print(f"输入 Tgt 形状: {tgt.shape}")
    print(f"Src Mask 形状: {src_mask.shape}")
    print(f"Tgt Mask 形状: {tgt_mask.shape}")

    # --- 前向传播 ---
    with torch.inference_mode():
        output = model(src, tgt, src_mask, tgt_mask)

    print(f"\n模型输出 Logits 形状: {output.shape}")
    print(f"预期形状: [Batch={batch_size}, Tgt_Len={tgt_len}, Tgt_Vocab={tgt_vocab_size}]")

    # 验证输出：Decoder 输出对应每个 tgt 位置的下一个 token 预测
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size)
    print("\n✅ Transformer 测试通过！")


if __name__ == "__main__":
    main()
