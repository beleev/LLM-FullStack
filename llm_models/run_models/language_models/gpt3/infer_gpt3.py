#!/usr/bin/env python
"""
GPT-3 模型示例

演示 GPT-3（Decoder-only Transformer）架构的使用方法。

Decoder-only 架构特点：
- 没有独立 Encoder，所有 token 在同一个序列里做自回归建模
- 用因果掩码（causal mask）保证位置 t 看不到 t+1..T-1（防偷看未来）
- 训练目标：next-token prediction（预测下一个 token）
- 输出 logits 形状 [B, T, V]，第 t 个位置的 logits 用于预测第 t+1 个 token

对应论文：Language Models are Few-Shot Learners (Brown et al., 2020)
"""

import torch
from llm_models.models import GPT3


def main():
    torch.manual_seed(42)

    # --- 演示用配置 (GPT-Mini) ---
    # 真实 GPT-3 是 175B 参数，这里用极小尺寸只为演示 API 与形状
    config = {
        "vocab_size": 1000,
        "d_model": 512,
        "n_heads": 8,
        "num_layers": 4,
        "max_len": 128,
        "dropout": 0.1,
        "use_rope": False  # False = 用经典 Sinusoidal 位置编码；True = RoPE
    }

    print("=" * 50)
    print("GPT-3 模型测试 (Demo Version)")
    print("=" * 50)
    print(f"词表大小: {config['vocab_size']}")
    print(f"模型维度: {config['d_model']}")
    print(f"注意力头数: {config['n_heads']}")
    print(f"层数: {config['num_layers']}")
    print(f"位置编码: {'RoPE' if config['use_rope'] else 'Sinusoidal'}")

    # 初始化模型
    print("\n正在初始化 GPT-3...")
    model = GPT3(**config)
    model.eval()  # Demo 只做推理：关闭 dropout

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 模拟输入：随机 token id 序列
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    print(f"\n输入形状: {input_ids.shape}")

    # 前向传播 (inference_mode: 跳过 autograd 版本计数，比 no_grad 略快)
    with torch.inference_mode():
        logits = model(input_ids)

    print(f"输出 Logits 形状: {logits.shape}")
    print(f"预期形状: [Batch={batch_size}, Seq_Len={seq_len}, Vocab={config['vocab_size']}]")

    # 简易自回归生成：取最后一个位置的 logits → argmax → 即下一个 token 预测
    # 真实生成时通常用 sampling / beam search，这里只是看模型能跑通生成接口
    next_token_pred = torch.argmax(logits[:, -1, :], dim=-1)
    print(f"\n下一个 token 预测: {next_token_pred.tolist()}")

    # 验证输出形状
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])
    print("\n✅ GPT-3 测试通过！")


if __name__ == "__main__":
    main()
