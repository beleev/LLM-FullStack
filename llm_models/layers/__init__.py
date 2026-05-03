"""
基础层 (layers) 模块 — Transformer 全栈零件库

按功能分类成四个子包:

  core/        通用 Transformer 基础组件 (任何 Transformer 都需要)
                - attention:     SDPA → MHA → GQA → MLA → DSA
                - position:      Sinusoidal / RoPE / M-RoPE
                - feedforward:   ReLU / GELU / SwiGLU
                - normalization: RMSNorm
                - blocks:        PreLN Transformer Block 组装器

  sparse/      稀疏 / 非注意力序列分支
                - moe: MixtralMoE (经典稀疏 MoE)
                - ssm: SelectiveSSM (Mamba)

  diffusion/   扩散 / 生成模型专用
                - adaln: adaLN-Zero (DiT 条件注入)
                - vq:    VectorQuantizer (VQ-VAE / VAR)

  multimodal/  跨模态构建块
                - PatchEmbed2D / 3D, PerceiverResampler, ModalityProjector

通过组合这些零件可拼出 BERT / GPT-3 / LLaMA / Mixtral / Qwen2-VL / DeepSeek-V3 /
Mamba / CLIP / Whisper / DiT / MM-DiT / Video DiT / VAR 等模型。
"""

from llm_models.layers.core import (
    ScaledDotProductAttention,
    SingleHeadSelfAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    MultiHeadLatentSparseAttention,
    SinPositionalEncoding,
    RotaryPositionalEncoding,
    MultimodalRotaryEmbedding,
    apply_rotary_pos_emb,
    sinusoidal_embedding,
    FeedForward,
    GeLUFeedForward,
    SwiGLUFeedForward,
    RMSNorm,
    PreLNBlock,
    PreLNCrossBlock,
)

from llm_models.layers.sparse import MixtralMoE, SelectiveSSM

from llm_models.layers.diffusion import (
    AdaLNZeroBlock,
    FinalLayer,
    TimestepEmbedding,
    modulate,
    VectorQuantizer,
)

from llm_models.layers.multimodal import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
)

__all__ = [
    # core/attention
    "ScaledDotProductAttention",
    "SingleHeadSelfAttention",
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "MultiHeadLatentSparseAttention",
    # core/position
    "SinPositionalEncoding",
    "RotaryPositionalEncoding",
    "MultimodalRotaryEmbedding",
    "apply_rotary_pos_emb",
    "sinusoidal_embedding",
    # core/feedforward
    "FeedForward",
    "GeLUFeedForward",
    "SwiGLUFeedForward",
    # core/normalization
    "RMSNorm",
    # core/blocks
    "PreLNBlock",
    "PreLNCrossBlock",
    # sparse
    "MixtralMoE",
    "SelectiveSSM",
    # diffusion
    "AdaLNZeroBlock",
    "FinalLayer",
    "TimestepEmbedding",
    "modulate",
    "VectorQuantizer",
    # multimodal
    "PatchEmbed2D",
    "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock",
    "PerceiverResampler",
    "ModalityProjector",
]
