#!/usr/bin/env python
"""
Qwen2-VL 架构演示示例

Qwen2-VL 是一个视觉语言模型（VLM = Vision Language Model）：
图像与文本 token 拼接后送入同一个 LLM，做多模态理解 / 文本生成。

本示例展示：
- 图像 + 文本的拼接输入：图像经 ViT 编码后，由 resampler 压缩为固定数量
  的视觉 token，与文本 token 在序列维度拼接
- 视觉 tokens 的 resampler 压缩（Perceiver/Q-Former 风格，固定 num_latents）
- 输出 logits 的形状对比：多模态 vs 纯文本（验证视觉 token 是否被加入序列）
"""

import torch
from llm_models.models import Qwen2VLModel


def main():
    torch.manual_seed(42)

    # --- Demo 配置（迷你尺寸，仅做形状演示） ---
    config = {
        "vocab_size": 1000,
        "text_d_model": 256,
        "text_n_heads": 8,
        "text_num_layers": 2,
        "max_len": 256,
        # ---- Vision encoder ----
        "vision_image_size": 224,
        "vision_patch_size": 16,         # 224/16 = 14 → 共 14*14 = 196 个 patch
        "vision_d_model": 256,
        "vision_n_heads": 8,
        "vision_num_layers": 2,
        "vision_num_latents": 32,        # Resampler 把 196 个 patch 压成 32 个 latent token
        "vision_num_latent_layers": 2,
        "projector_hidden_dim": 512,     # 投影到文本 d_model 的中间隐层维度
        "dropout": 0.1,
        "use_rope": True,
        "use_modality_embedding": True,  # 给视觉/文本 token 加模态区分嵌入
    }

    print("=" * 60)
    print("Qwen2-VL 架构演示 (Demo Mode)")
    print("=" * 60)
    for k, v in config.items():
        print(f"{k}: {v}")

    # 初始化模型
    print("\n正在初始化 Qwen2-VL 模型...")
    model = Qwen2VLModel(**config)
    model.eval()  # Demo 只做推理：关闭 dropout

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params:,}")

    # --- 构造模拟输入 ---
    batch_size = 2
    text_len = 16
    image_size = config["vision_image_size"]

    # 文本 token IDs（随机词表索引）
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, text_len))

    # 图像输入：标准 (B, C, H, W)
    images = torch.randn(batch_size, 3, image_size, image_size)

    # 文本 padding mask (全部有效，无 pad 位置)
    text_attention_mask = torch.ones(batch_size, text_len, dtype=torch.bool)

    # --- 前向传播 (多模态：图像 + 文本) ---
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            images=images,
            text_attention_mask=text_attention_mask,
        )

    print("\n多模态输入输出:")
    print(f"input_ids: {input_ids.shape}")
    print(f"images: {images.shape}")
    print(f"logits: {logits.shape}")

    # --- 前向传播 (纯文本，验证模型对单模态输入也兼容) ---
    with torch.inference_mode():
        logits_text_only = model(input_ids=input_ids)

    print("\n纯文本输入输出:")
    print(f"input_ids: {input_ids.shape}")
    print(f"logits: {logits_text_only.shape}")

    # --- 形状校验 ---
    # 多模态：序列 = 视觉 latent token + 文本 token
    expected_seq_len = text_len + config["vision_num_latents"]
    assert logits.shape == (batch_size, expected_seq_len, config["vocab_size"])
    # 纯文本：序列只有文本 token
    assert logits_text_only.shape == (batch_size, text_len, config["vocab_size"])

    print("\n✅ Qwen2-VL 架构演示通过！")


if __name__ == "__main__":
    main()
