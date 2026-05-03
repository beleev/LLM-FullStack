#!/usr/bin/env python
"""
DeepSeek-V3 模型示例

演示 DeepSeek-V3（MoE，Mixture-of-Experts 架构）的使用方法。

MoE 核心思想：
- 把传统 Transformer 中的单一 FFN 替换为多个"专家"（小 FFN）
- 每个 token 只激活其中 top-k 个专家（稀疏激活），其余专家不参与计算
- 总参数量很大，但每个 token 实际激活的参数量很少 → 算力成本低
- 共享专家：每个 token 都会经过的"通用专家"，承载共有知识
- 路由专家：由 router（门控网络）选出 top-k 的"专用专家"

对应论文：DeepSeek-V3 Technical Report
"""

import torch
from llm_models.models import DeepSeekV3


def main():
    torch.manual_seed(42)

    # --- DeepSeek-V3 Demo 配置 ---
    # 极小规模，仅用于演示 API 与 MoE 路由信息
    config = {
        "vocab_size": 1000,
        "d_model": 512,
        "n_heads": 8,
        "num_layers": 2,
        "num_routed_experts": 8,   # 路由专家池大小
        "num_shared_experts": 1,   # 始终激活的共享专家数
        "top_k": 2,                # 每个 token 从路由专家池选 top-2
        "dropout": 0.1
    }

    print("=" * 50)
    print("DeepSeek-V3 模型测试 (Demo Mode)")
    print("=" * 50)
    print(f"词表大小: {config['vocab_size']}")
    print(f"模型维度: {config['d_model']}")
    print(f"注意力头数: {config['n_heads']}")
    print(f"层数: {config['num_layers']}")
    print(f"\nMoE 配置:")
    print(f"  共享专家数: {config['num_shared_experts']}")
    print(f"  路由专家数: {config['num_routed_experts']}")
    print(f"  Top-K: {config['top_k']}")
    print(f"\n策略: {config['num_shared_experts']} 共享专家 + "
          f"Top-{config['top_k']} from {config['num_routed_experts']} 路由专家")

    # 初始化模型
    print("\n正在初始化 DeepSeek-V3...")
    model = DeepSeekV3(**config)
    model.eval()  # Demo 只做推理：关闭 dropout

    # 打印参数量（注意：MoE 总参数 ≠ 单 token 实际激活参数）
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params:,}")

    # 模拟输入：随机 token id
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))

    print(f"\n输入形状: {input_ids.shape}")

    # 前向传播；除了 logits 还返回每层的 routing 信息，用于观察 MoE 行为
    with torch.inference_mode():
        logits, all_routing_info = model(input_ids)

    print(f"输出 Logits 形状: {logits.shape}")
    print(f"预期形状: [Batch={batch_size}, Seq_Len={seq_len}, Vocab={config['vocab_size']}]")

    # 验证输出形状：标准 LM 头输出 = [B, T, vocab]
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])

    # 展示 MoE 路由信息（每个 token 选择了哪些专家、权重多少）
    # 这一步是教学演示的关键 —— 看到 router 对不同 token 做出了不同选择
    print(f"\n--- MoE 路由信息 ---")
    for i, routing_info in enumerate(all_routing_info):
        selected = routing_info["selected_experts"]    # [B, T, top_k] 选中的专家 id
        weights = routing_info["routing_weights"]      # [B, T, top_k] 对应权重（softmax 后）
        print(f"  Layer {i}: 选中专家 {selected[0].tolist()}, "
              f"权重 {weights[0].round(decimals=3).tolist()}")

    # 展示活跃参数量：MoE 的核心卖点 —— 总参数大、单 token 激活参数小
    param_info = model.get_num_active_params()
    print(f"\n--- 参数量统计 ---")
    print(f"  总参数量: {param_info['total_params']:,}")
    print(f"  每 token 激活参数量: {param_info['active_params']:,}")
    print(f"  激活比例: {param_info['active_params'] / param_info['total_params']:.1%}")

    print("\n✅ DeepSeek-V3 (MoE Architecture) 测试通过！")


if __name__ == "__main__":
    main()
