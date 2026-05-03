#!/usr/bin/env python
"""
DeepSeek-V3.2 模型示例

演示 DeepSeek-V3.2 的使用方法。在 V3 (MoE) 基础上引入：
- DSA (DeepSeek Sparse Attention)：稀疏注意力，每个 query 只与
  top-k 个 key 计算注意力，把 O(T^2) 复杂度降到 O(T·k)，
  显著加快长序列推理。
- MLA (Multi-head Latent Attention)：将 KV 投影到低维 latent，
  压缩 KV cache 占用。
- MoE 部分继承自 V3。

DSA 通过一个轻量 indexer（小注意力头）打分，选出每个 query 最相关的
sparse_top_k 个 key 参与正式注意力。
"""

import torch
from llm_models.models import DeepSeekV3_2


def main():
    torch.manual_seed(42)

    # --- DeepSeek-V3.2 Demo 配置 ---
    config = {
        "vocab_size": 1000,
        "d_model": 512,
        "n_heads": 8,
        "num_layers": 2,
        "num_routed_experts": 8,
        "num_shared_experts": 1,
        "top_k": 2,
        "dropout": 0.1,
        # DSA 配置
        "sparse_top_k": 16,    # 每个 query 选择参与 attention 的 key 数
        "indexer_heads": 2,    # indexer 用多少个轻量头去打分选 key
    }

    print("=" * 50)
    print("DeepSeek-V3.2 模型测试 (Demo Mode)")
    print("=" * 50)
    print(f"词表大小: {config['vocab_size']}")
    print(f"模型维度: {config['d_model']}")
    print(f"注意力头数: {config['n_heads']}")
    print(f"层数: {config['num_layers']}")
    print(f"\nMoE 配置:")
    print(f"  共享专家数: {config['num_shared_experts']}")
    print(f"  路由专家数: {config['num_routed_experts']}")
    print(f"  Top-K: {config['top_k']}")
    print(f"\nDSA 配置:")
    print(f"  稀疏 Top-K: {config['sparse_top_k']}")
    print(f"  Indexer 头数: {config['indexer_heads']}")

    # 初始化模型
    print("\n正在初始化 DeepSeek-V3.2...")
    model = DeepSeekV3_2(**config)
    model.eval()  # Demo 只做推理：关闭 dropout

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params:,}")

    # 模拟输入
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    print(f"\n输入形状: {input_ids.shape}")

    # 前向传播 (返回 logits 和每层的 MoE 路由信息)
    with torch.inference_mode():
        logits, all_routing_info = model(input_ids)

    print(f"输出 Logits 形状: {logits.shape}")
    print(f"预期形状: [Batch={batch_size}, Seq_Len={seq_len}, Vocab={config['vocab_size']}]")

    # 验证输出形状
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])

    # 展示 MoE 路由信息 (观测专家选择，验证路由没有崩塌到固定专家)
    print(f"\n--- MoE 路由信息 ---")
    for i, routing_info in enumerate(all_routing_info):
        selected = routing_info["selected_experts"]
        weights = routing_info["routing_weights"]
        print(f"  Layer {i}: 选中专家 {selected[0].tolist()}, "
              f"权重 {weights[0].round(decimals=3).tolist()}")

    print("\n✅ DeepSeek-V3.2 (DSA + MoE) 测试通过！")


if __name__ == "__main__":
    main()
