#!/usr/bin/env python
"""
Qwen2.5-Omni (多模态) 训练示例

演示使用通用 Trainer 训练 Qwen2.5-Omni。
使用合成数据 (图像 + 音频 + 视频 + 文本) 进行联合训练。

损失定义: total_loss = text_loss + audio_loss_weight * audio_loss
- text_loss:  Thinker 输出的文本 logits 上的交叉熵
- audio_loss: Talker 输出的音频 token logits 上的交叉熵
- audio_loss_weight 用来平衡两个任务的相对权重

合成数据按真实模型预期形状随机构造，仅做训练循环验证，不会得到有意义模型。
"""

import torch
from llm_models.models import Qwen2_5_OmniModel
from llm_models.training import (
    Trainer,
    TrainingConfig,
    OmniLoss,
    OmniDataGenerator,
)


def main():
    # --- 训练配置 ---
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=1,                # 多模态模型显存/算力更重，batch 用 1
        seq_len=8,
        num_steps=50,
        warmup_steps=5,
        audio_loss_weight=0.5,       # text/audio loss 平衡因子
        log_interval=10,
        seed=42,
    )
    torch.manual_seed(config.seed)

    # --- 模型配置 (Omni-Tiny) ---
    # 各模态 latent token 数（resampler 输出维度）需与 data_gen 保持一致
    vocab_size = 500
    audio_vocab_size = 200           # 离散语音 token 词表（Talker 输出）
    image_size = 56
    audio_spec_size = (64, 32)
    video_size = (4, 56, 56)         # (T, H, W) 4 帧

    # 各编码器经 resampler 后产出的固定 token 数
    num_vision_latents = 8
    num_audio_latents = 8
    num_video_latents = 8

    model = Qwen2_5_OmniModel(
        vocab_size=vocab_size,
        audio_vocab_size=audio_vocab_size,
        text_d_model=128,
        text_n_heads=4,
        text_num_layers=2,
        max_len=256,
        # Vision
        vision_image_size=image_size,
        vision_patch_size=14,
        vision_d_model=128,
        vision_n_heads=4,
        vision_num_layers=1,
        vision_num_latents=num_vision_latents,
        vision_num_latent_layers=1,
        # Audio
        audio_spec_size=audio_spec_size,
        audio_patch_size=(8, 8),
        audio_in_channels=1,
        audio_d_model=64,
        audio_n_heads=4,
        audio_num_layers=1,
        audio_num_latents=num_audio_latents,
        audio_num_latent_layers=1,
        # Video
        video_size=video_size,
        video_tubelet_size=2,
        video_patch_size=14,
        video_in_channels=3,
        video_d_model=128,
        video_n_heads=4,
        video_num_layers=1,
        video_num_latents=num_video_latents,
        video_num_latent_layers=1,
        # Talker (音频生成 decoder)
        talker_n_heads=4,
        talker_num_layers=1,
        talker_max_len=128,
        # Misc
        dropout=0.1,
        use_rope=False,
        use_modality_embedding=True,  # 模态嵌入帮助模型区分输入来源
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Qwen2.5-Omni Tiny | 参数量: {num_params:,}")

    # --- 数据生成器 + 损失函数 ---
    # OmniDataGenerator 同步生成各模态合成输入及 text/audio 两套 label
    # num_*_tokens 必须与模型 num_*_latents 对齐，否则序列长度对不上
    data_gen = OmniDataGenerator(
        vocab_size=vocab_size,
        audio_vocab_size=audio_vocab_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        audio_seq_len=8,
        image_size=image_size,
        audio_spec_size=audio_spec_size,
        video_size=video_size,
        num_vision_tokens=num_vision_latents,
        num_audio_tokens=num_audio_latents,
        num_video_tokens=num_video_latents,
    )
    loss_fn = OmniLoss(audio_loss_weight=config.audio_loss_weight)

    # --- 训练 ---
    trainer = Trainer(model, config, data_gen, loss_fn)
    metrics = trainer.train()

    # --- 验证 ---
    assert metrics[-1]["total_loss"] < metrics[0]["total_loss"], "Loss 未下降!"
    print(f"最终 text_loss: {metrics[-1]['text_loss']:.4f} | "
          f"audio_loss: {metrics[-1]['audio_loss']:.4f}")
    print("Qwen2.5-Omni 训练验证通过!")


if __name__ == "__main__":
    main()
