"""
Whisper 模型模块

论文出处:
    "Robust Speech Recognition via Large-Scale Weak Supervision"
    (Radford et al., OpenAI, 2022)

在本库中的位置:
    与原始 Encoder-Decoder Transformer (2017) 同架构家族, 但:
        - Encoder 输入不是 token 而是 **log-mel spectrogram** (80 维频谱)
        - 前置 2 层 1D conv stem: 把 spectrogram 压到适合 Transformer 的长度
        - 任务: 语音识别 / 翻译 / 语言识别, 通过 task token 切换
    与 Qwen2.5-Omni 的音频路径对比:
        Omni: mel → PatchEmbed2D → PatchTransformerEncoder → Resampler → LLM 前缀
        Whisper: mel → Conv stem → 6-32 层 encoder → 交叉注意力喂给 decoder

教学重点:
    - 语音建模 **复用 Transformer 全部机器** 的同时, 只替换了 "token 化" 步骤:
        Conv1d(kernel=3, stride=2) ×2 把时间维压到约 50 Hz 即可做 Attention
    - Decoder 与经典 Transformer decoder 完全相同, 用交叉注意力读 encoder 输出
    - 用 sinusoidal 绝对位置 (encoder 侧) + learned 位置 (decoder 侧), Whisper 官方做法
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from llm_models.layers.core.attention import MultiHeadAttention
from llm_models.layers.core.blocks import PreLNBlock, PreLNCrossBlock
from llm_models.layers.core.feedforward import GeLUFeedForward
from llm_models.layers.core.position_encoding import SinPositionalEncoding
from llm_models.utils.masks import build_causal_mask


class WhisperAudioEncoder(nn.Module):
    """
    Whisper 音频 encoder

    数据流:
        mel [B, n_mels, T_mel]
           -> Conv1d(n_mels → d_model, kernel=3, stride=1, padding=1) + GELU
           -> Conv1d(d_model → d_model, kernel=3, stride=2, padding=1) + GELU
           (时间维缩 2×, 对 Whisper 输入 30s / 16kHz / hop=10ms 即 3000 帧 → 1500 帧)
           -> 加 Sinusoidal 位置编码
           -> N x PreLNBlock(MHA + GELU-FFN + LayerNorm)
           -> LayerNorm
        output: [B, T_enc, d_model]

    Args:
        n_mels:       mel 滤波器组数 (Whisper 用 80)
        d_model:      隐藏维度 (Whisper-base 512, large 1280)
        n_heads:      注意力头数
        num_layers:   encoder 层数
        max_source_len: 预构建位置编码的最长帧数 (1500 对应 30s@50Hz)
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        max_source_len: int = 1500,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 2 层 1D conv stem: 把 mel-spectrogram 变成 token 序列
        # stride=2 仅在第二层, 整体时间维 2× 下采样
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # 正弦位置编码覆盖整个 encoder 长度
        self.pos_encoding = SinPositionalEncoding(d_model, max_len=max_source_len)

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [
                PreLNBlock(
                    d_model=d_model,
                    attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T_mel]  log-mel 声谱图
        Returns:
            [B, T_enc, d_model]  encoder 输出, 供 decoder 交叉注意力查询
        """
        x = torch.nn.functional.gelu(self.conv1(mel))
        x = torch.nn.functional.gelu(self.conv2(x))                    # [B, d_model, T/2]
        x = x.transpose(1, 2)                                           # [B, T/2, d_model]

        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)                                                # encoder 双向 attn
        return self.ln_f(x)


class WhisperTextDecoder(nn.Module):
    """
    Whisper 文本 decoder: 标准因果 self-attn + cross-attn(encoder) + FFN

    与原始 Transformer decoder 完全同构, 这里直接复用 PreLNCrossBlock。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        max_target_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_target_len = max_target_len

        # Whisper decoder 用 learned 位置嵌入 (而非 encoder 的 sinusoidal)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_target_len, d_model)

        d_ff = 4 * d_model
        self.layers = nn.ModuleList(
            [
                PreLNCrossBlock(
                    d_model=d_model,
                    self_attn=MultiHeadAttention(d_model, n_heads),
                    cross_attn=MultiHeadAttention(d_model, n_heads),
                    ffn=GeLUFeedForward(d_model, d_ff),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        # lm_head 与 embedding 共享权重
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        causal = build_causal_mask(max_target_len, torch.device("cpu"))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, T_tgt] decoder 输入 token (含 task prompt)
            encoder_hidden: [B, T_enc, d_model] encoder 输出
        Returns:
            logits: [B, T_tgt, vocab_size]
        """
        B, T = input_ids.shape
        if T > self.max_target_len:
            raise ValueError(f"target 长度 {T} 超过 max_target_len={self.max_target_len}")

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)

        self_mask = self.causal_mask[:, :T, :T]

        for layer in self.layers:
            x = layer(
                x,
                context=encoder_hidden,
                self_mask=self_mask,
                context_mask=None,
            )
        x = self.ln_f(x)
        return self.lm_head(x)


class Whisper(nn.Module):
    """
    Whisper 完整模型 (教学版)

    架构:
        mel  -> WhisperAudioEncoder   -> hidden_enc
        tgt  -> WhisperTextDecoder(cross-attend hidden_enc) -> logits

    Args 见 encoder / decoder 的 docstring。
    """

    def __init__(
        self,
        vocab_size: int = 51865,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        max_source_len: int = 1500,
        max_target_len: int = 448,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = WhisperAudioEncoder(
            n_mels=n_mels, d_model=d_model, n_heads=n_heads,
            num_layers=encoder_layers, max_source_len=max_source_len, dropout=dropout,
        )
        self.decoder = WhisperTextDecoder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            num_layers=decoder_layers, max_target_len=max_target_len, dropout=dropout,
        )

    def forward(
        self,
        mel: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T_mel]
            decoder_input_ids: [B, T_tgt]
        Returns:
            logits: [B, T_tgt, vocab_size]
        """
        enc = self.encoder(mel)
        return self.decoder(decoder_input_ids, encoder_hidden=enc)
