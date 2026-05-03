#!/usr/bin/env python
"""
注意力机制示例

演示 MultiHeadAttention 的最小用法：
- 构造一个随机 [B, T, D] 输入
- 让多头注意力做一次自注意力（Self-Attention）前向
- 验证输出形状与输入一致（这是 Attention 层"残差兼容"的关键性质，
  因为 Transformer block 会用 residual connection 把输入加回输出）

对应论文：Attention Is All You Need (Vaswani et al., 2017)
"""

import torch
from llm_models.layers import MultiHeadAttention


def main():
    # 固定随机种子，保证演示可复现
    torch.manual_seed(42)

    # ================= 配置参数 =================
    # 这些值都很小，纯粹为了 CPU 上秒级跑通
    batch_size = 2     # 一个 batch 中的句子数
    n_heads = 4        # 注意力头数；d_model 必须能被它整除
    seq_len = 16       # 序列长度（token 数）
    d_model = 128      # 每个 token 的特征维度；每个 head 维度 = 128/4 = 32
    # ===========================================

    print(f"--- 模拟场景: Attention 处理 {batch_size} 个句子 ---")
    print(f"    Head 数量: {n_heads}")
    print(f"    序列长度: {seq_len}")
    print(f"    模型维度: {d_model}")

    # 1. 创建随机输入数据（这里用 randn 模拟"已经过 embedding"的特征）
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    print(f"\n1. 输入形状 (Input): {input_tensor.shape}")
    print("   (Batch_Size, Seq_Len, D_Model)")

    # 2. 实例化模型
    self_attn_layer = MultiHeadAttention(d_model, n_heads)
    self_attn_layer.eval()  # Demo 只做推理：关闭 dropout

    # 3. 前向传播 (Forward)
    # inference_mode 比 no_grad 更激进：跳过 autograd 版本计数，速度更快
    with torch.inference_mode():
        # 单参数调用 = self-attention：Q=K=V=input_tensor
        output_tensor = self_attn_layer(input_tensor)

    # 4. 查看结果
    print(f"\n2. 输出形状 (Output): {output_tensor.shape}")
    print("   (注意：输出形状与输入完全一致，方便堆叠多层)")

    # 5. 验证：形状不变是 Attention 层最基本的契约
    assert output_tensor.shape == input_tensor.shape
    print("\n✅ MultiHeadAttention 测试通过！")


if __name__ == "__main__":
    main()
