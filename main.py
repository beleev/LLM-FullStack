#!/usr/bin/env python
"""
LLM Models - 主入口文件

模块化大语言模型教学库的总览入口:
1. 列出当前库支持的架构 (左脑 LLM / 多模态理解 / 右脑生成)
2. 列出可直接运行的示例脚本
3. 用几个最小前向做 "冒烟测试", 确保依赖 & 模块导入都正确

并非真正的训练入口; 具体训练/推理示例见 llm_models.examples。
"""

import torch

from llm_models import (
    MultiHeadAttention,
    GPT3,
    LLaMA,
    Mixtral,
    Mamba,
    BERT,
    CLIPModel,
    Whisper,
    DiT,
    ImageVAE,
    DDPMScheduler,
)


def _smoke_attention(n_params_fn) -> None:
    torch.manual_seed(42)
    attn = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(1, 10, 64)
    out = attn(x)
    print(f"  MHA:     输入 {tuple(x.shape)} -> 输出 {tuple(out.shape)}  ✓ ")


def _smoke_gpt() -> None:
    m = GPT3(vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32).eval()
    with torch.inference_mode():
        logits = m(torch.randint(0, 100, (1, 8)))
    print(f"  GPT-3:   logits {tuple(logits.shape)}  ✓")


def _smoke_llama() -> None:
    m = LLaMA(vocab_size=100, d_model=64, n_heads=4, num_kv_heads=2,
              num_layers=2, max_len=32).eval()
    with torch.inference_mode():
        logits = m(torch.randint(0, 100, (1, 8)))
    print(f"  LLaMA:   logits {tuple(logits.shape)}  ✓")


def _smoke_mixtral() -> None:
    m = Mixtral(vocab_size=100, d_model=64, n_heads=4, num_kv_heads=2,
                num_layers=2, num_experts=4, top_k=2, max_len=32).eval()
    with torch.inference_mode():
        logits, routing = m(torch.randint(0, 100, (1, 8)))
    print(f"  Mixtral: logits {tuple(logits.shape)}, routing 层数 {len(routing)}  ✓")


def _smoke_mamba() -> None:
    m = Mamba(vocab_size=100, d_model=64, num_layers=2).eval()
    with torch.inference_mode():
        logits = m(torch.randint(0, 100, (1, 8)))
    print(f"  Mamba:   logits {tuple(logits.shape)}  ✓")


def _smoke_bert() -> None:
    m = BERT(vocab_size=100, d_model=64, n_heads=4, num_layers=2, max_len=32).eval()
    with torch.inference_mode():
        logits = m(torch.randint(0, 100, (1, 8)))
    print(f"  BERT:    logits {tuple(logits.shape)}  ✓")


def _smoke_clip() -> None:
    m = CLIPModel(
        embed_dim=64, vocab_size=100,
        text_d_model=64, text_n_heads=4, text_num_layers=2, text_max_len=16,
        image_size=32, patch_size=8,
        vision_d_model=64, vision_n_heads=4, vision_num_layers=2,
    ).eval()
    with torch.inference_mode():
        out = m(torch.randn(2, 3, 32, 32), torch.randint(0, 100, (2, 8)))
    print(f"  CLIP:    img {tuple(out['image_features'].shape)} "
          f"txt {tuple(out['text_features'].shape)}  ✓")


def _smoke_whisper() -> None:
    m = Whisper(vocab_size=100, n_mels=80, d_model=64, n_heads=4,
                encoder_layers=1, decoder_layers=1,
                max_source_len=50, max_target_len=16).eval()
    with torch.inference_mode():
        logits = m(torch.randn(1, 80, 20), torch.randint(0, 100, (1, 4)))
    print(f"  Whisper: logits {tuple(logits.shape)}  ✓")


def _smoke_dit() -> None:
    m = DiT(latent_channels=4, image_size=4, patch_size=2,
            d_model=32, n_heads=4, num_layers=2, num_classes=2).eval()
    scheduler = DDPMScheduler(num_train_timesteps=100)
    x0 = torch.randn(1, 4, 4, 4)
    t = scheduler.sample_timesteps(1, x0.device)
    noised = scheduler.add_noise(x0, t)
    with torch.inference_mode():
        pred = m(noised.noisy, noised.t_norm, torch.tensor([0]))
    print(f"  DiT:     x_t {tuple(noised.noisy.shape)} -> ε̂ {tuple(pred.shape)}  ✓")


def _smoke_vae() -> None:
    m = ImageVAE(base_channels=16, levels=1).eval()
    with torch.inference_mode():
        out = m(torch.randn(1, 3, 16, 16))
    print(f"  VAE:     recon {tuple(out['recon'].shape)} z {tuple(out['z'].shape)}  ✓")


def main():
    print("=" * 70)
    print("LLM Models - 模块化大语言模型 & 生成模型教学库")
    print("=" * 70)

    print("\n[左脑] 语言理解 / 文本生成:")
    print("  Transformer | BERT | GPT-3 | LLaMA | Mixtral | Mamba | "
          "DeepSeek-V3 / V3.2")

    print("\n[眼耳] 多模态理解:")
    print("  CLIP | Whisper | Qwen2-VL | Qwen2.5-Omni")

    print("\n[右脑] 生成模型:")
    print("  ImageVAE | CausalVideoVAE | DiT | MM-DiT (SD3) | Video DiT (Sora) | VAR")

    print("\n示例脚本 (examples/):")
    print("  run_attention / run_bert / run_transformer / run_gpt3")
    print("  run_llama / run_mixtral / run_mamba")
    print("  run_deepseek / run_deepseek_v3_2")
    print("  run_clip / run_whisper")
    print("  run_qwen2_vl_demo / run_qwen2_5_omni_demo")
    print("  run_vae / run_dit / run_video_dit / run_mmdit / run_var")
    print("  train_* 对应上述所有模型的合成数据训练验证")

    print("\n冒烟测试 (每个核心模型跑一次最小前向):")
    _smoke_attention(lambda m: sum(p.numel() for p in m.parameters()))
    _smoke_bert()
    _smoke_gpt()
    _smoke_llama()
    _smoke_mixtral()
    _smoke_mamba()
    _smoke_clip()
    _smoke_whisper()
    _smoke_vae()
    _smoke_dit()

    print("\n安装:  pip install -e .")
    print("更多: 见 README.md")


if __name__ == "__main__":
    main()
