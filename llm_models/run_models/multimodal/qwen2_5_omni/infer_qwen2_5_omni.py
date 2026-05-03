#!/usr/bin/env python
"""
Qwen2.5-Omni 架构演示示例

Qwen2.5-Omni 是一个全模态（Omni-modal）模型，能同时处理
图像 / 视频 / 音频（声谱图）+ 文本输入，并产出文本与音频两种输出。

本示例展示：
- 图像/视频/声谱图 + 文本的多模态拼接输入流程
- Thinker：负责"理解"与文本生成，输出文本 logits
- Talker：基于 Thinker 的 hidden states 生成语音 token 序列（语音 logits）

各模态编码器都使用 Resampler（Perceiver 风格）把可变数量的 patch
压缩到固定数量的 latent token（num_latents），再拼到文本序列前。
这样无论输入图像/视频多大，进入 LLM 的视觉 token 数都是固定的。
"""

import torch
from llm_models.models import Qwen2_5_OmniModel


def main():
    torch.manual_seed(42)

    # 极小的 Demo 配置，纯粹为了 CPU 跑通；真实模型每个维度大几十倍
    config = {
        "vocab_size": 1000,
        "audio_vocab_size": 512,        # 语音离散 token 词表（用于 Talker 输出）
        "text_d_model": 256,
        "text_n_heads": 8,
        "text_num_layers": 2,
        "max_len": 256,
        # ---- Vision encoder（处理静态图像） ----
        "vision_image_size": 64,
        "vision_patch_size": 16,
        "vision_d_model": 256,
        "vision_n_heads": 8,
        "vision_num_layers": 2,
        "vision_num_latents": 8,        # Resampler 输出的视觉 token 数（固定）
        "vision_num_latent_layers": 2,
        # ---- Audio encoder（处理声谱图） ----
        "audio_spec_size": (64, 32),    # (freq, time)
        "audio_patch_size": (8, 8),
        "audio_in_channels": 1,
        "audio_d_model": 128,
        "audio_n_heads": 4,
        "audio_num_layers": 2,
        "audio_num_latents": 8,
        "audio_num_latent_layers": 2,
        # ---- Video encoder（tubelet = 时空 patch） ----
        "video_size": (4, 64, 64),      # (T, H, W) 4 帧
        "video_tubelet_size": 2,        # 每 2 帧合成一个 tubelet
        "video_patch_size": 16,
        "video_in_channels": 3,
        "video_d_model": 256,
        "video_n_heads": 8,
        "video_num_layers": 2,
        "video_num_latents": 8,
        "video_num_latent_layers": 2,
        # ---- Talker（语音生成 decoder） ----
        "talker_d_model": 256,
        "talker_n_heads": 8,
        "talker_num_layers": 2,
        "talker_max_len": 64,
        "dropout": 0.1,
        "use_rope": True,
        "use_modality_embedding": True,  # 给不同模态加可学习的模态嵌入，便于模型区分
    }

    print("=" * 60)
    print("Qwen2.5-Omni 架构演示 (Demo Mode)")
    print("=" * 60)
    for k, v in config.items():
        print(f"{k}: {v}")

    print("\n正在初始化 Qwen2.5-Omni 模型...")
    model = Qwen2_5_OmniModel(**config)
    model.eval()  # Demo 只做推理：关闭 dropout

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params:,}")

    # ---- 构造模拟输入 ----
    batch_size = 2
    text_len = 12
    audio_len = 20

    # 文本与音频 token id（音频 token 用于 Talker 的目标侧 / teacher forcing）
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, text_len))
    audio_input_ids = torch.randint(0, config["audio_vocab_size"], (batch_size, audio_len))

    # 各模态原始信号：图像 / 声谱图 / 视频
    images = torch.randn(batch_size, 3, config["vision_image_size"], config["vision_image_size"])
    audio_specs = torch.randn(batch_size, 1, *config["audio_spec_size"])
    videos = torch.randn(batch_size, 3, *config["video_size"])

    # padding mask（这里全部为 True 表示无 pad）
    text_attention_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    audio_attention_mask = torch.ones(batch_size, audio_len, dtype=torch.bool)

    # ---- 多模态前向 ----
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=images,
            audio_spectrograms=audio_specs,
            videos=videos,
            text_attention_mask=text_attention_mask,
            audio_input_ids=audio_input_ids,
            audio_attention_mask=audio_attention_mask,
            return_dict=True,
        )

    text_logits = outputs["text_logits"]    # Thinker 输出：用于文本生成
    audio_logits = outputs["audio_logits"]  # Talker 输出：用于语音 token 生成

    print("\n多模态输入输出:")
    print(f"input_ids: {input_ids.shape}")
    print(f"images: {images.shape}")
    print(f"audio_specs: {audio_specs.shape}")
    print(f"videos: {videos.shape}")
    print(f"text_logits: {text_logits.shape}")
    print(f"audio_logits: {audio_logits.shape}")

    # 实际进入 Thinker 的序列长度 = 各模态 latent token 数 + 文本 token 数
    # 因为 vision/video/audio 各自被 resampler 压缩成固定 num_latents 个 token
    expected_text_len = (
        config["vision_num_latents"]
        + config["video_num_latents"]
        + config["audio_num_latents"]
        + text_len
    )
    assert text_logits.shape == (batch_size, expected_text_len, config["vocab_size"])
    assert audio_logits.shape == (batch_size, audio_len, config["audio_vocab_size"])

    print("\n✅ Qwen2.5-Omni 架构演示通过！")


if __name__ == "__main__":
    main()
