#!/usr/bin/env python
"""
Qwen2-VL (视觉语言模型) 训练示例

演示使用通用 Trainer 训练 Qwen2-VL。
使用合成图像 + 文本数据，loss 仅在文本 token 位置计算
(视觉 token 位置的 label 用 -100 标记，CrossEntropyLoss 默认 ignore_index=-100
 → 这些位置不参与 loss 计算)。

为什么视觉位置不算 loss：
- 视觉 token 来自图像编码器，不存在"下一个 token 预测"的目标
- 模型的训练目标是基于"图像 + 已生成文本"预测下一个文本 token
- 因此只在文本位置上算 next-token loss，视觉位置只作为 condition
"""

import torch
from llm_models.models import Qwen2VLModel
from llm_models.training import (
    Trainer,
    TrainingConfig,
    StandardLMLoss,
    VisionLanguageDataGenerator,
)


def main():
    # --- 训练配置 ---
    config = TrainingConfig(
        learning_rate=1e-4,        # VL 模型用更小 lr，多模态训练更敏感
        batch_size=2,
        seq_len=16,
        num_steps=30,
        warmup_steps=3,
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(config.seed)

    # --- 模型配置 (VL-Mini) ---
    vocab_size = 500
    image_size = 56
    patch_size = 14                # 56/14 = 4 → 共 4*4 = 16 个 patch
    num_vision_latents = 16        # Resampler 输出 token 数（拼到文本前面）

    model = Qwen2VLModel(
        vocab_size=vocab_size,
        text_d_model=128,
        text_n_heads=4,
        text_num_layers=2,
        max_len=256,
        vision_image_size=image_size,
        vision_patch_size=patch_size,
        vision_d_model=128,
        vision_n_heads=4,
        vision_num_layers=2,
        vision_num_latents=num_vision_latents,
        vision_num_latent_layers=1,
        dropout=0.1,
        use_rope=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Qwen2-VL Mini | 参数量: {num_params:,}")

    # --- 数据生成器 + 损失函数 ---
    # VisionLanguageDataGenerator: 同步生成图像 + 文本对
    # 在 label 中，视觉 token 位置会被填 -100 → 自动从 loss 中剔除
    data_gen = VisionLanguageDataGenerator(
        vocab_size=vocab_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        image_size=image_size,
        num_vision_tokens=num_vision_latents,
    )
    # StandardLMLoss = CrossEntropy(ignore_index=-100)
    loss_fn = StandardLMLoss()

    # --- 训练 ---
    trainer = Trainer(model, config, data_gen, loss_fn)
    metrics = trainer.train()

    # --- 验证 ---
    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print("Qwen2-VL 训练验证通过!")


if __name__ == "__main__":
    main()
