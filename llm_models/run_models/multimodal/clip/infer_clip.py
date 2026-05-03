#!/usr/bin/env python
"""
CLIP 对比学习双塔前向示例

展示:
- 图像塔 + 文本塔独立编码, 各自产出 embed_dim 维向量
- 两塔特征 L2 归一化, 相似度 = 内积
- logit_scale 可学习温度
"""

import torch
from llm_models.models.multimodal.clip import CLIPModel


def main():
    torch.manual_seed(42)

    vocab_size = 1000
    model = CLIPModel(
        embed_dim=256, vocab_size=vocab_size,
        text_d_model=128, text_n_heads=4, text_num_layers=2, text_max_len=32,
        image_size=64, patch_size=16,
        vision_d_model=128, vision_n_heads=4, vision_num_layers=2,
    ).eval()
    print(f"CLIP Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    batch = 4
    images = torch.randn(batch, 3, 64, 64)
    text = torch.randint(1, vocab_size - 1, (batch, 16))
    text[:, -1] = vocab_size - 1   # EOS

    with torch.inference_mode():
        out = model(images, text, eos_token_id=vocab_size - 1)

    print(f"image_features: {tuple(out['image_features'].shape)}")
    print(f"text_features:  {tuple(out['text_features'].shape)}")
    print(f"logit_scale:    {out['logit_scale'].item():.3f}")

    # 构造相似度矩阵看对角线是否被正确对齐 (训练前就是随机)
    sim = out["image_features"] @ out["text_features"].t()
    print(f"相似度矩阵形状: {tuple(sim.shape)}")
    print("✅ CLIP 前向通过")


if __name__ == "__main__":
    main()
