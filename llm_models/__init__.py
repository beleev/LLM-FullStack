"""
LLM Models — 大语言模型架构演进教具库

PyTorch 实现, 用于对比各主流架构的设计与训练 / 推理数据流。

架构演进主线 (左脑: 理解 + 文本生成)
    Transformer (2017)      -> Encoder-Decoder + MHA + FFN + LN + Sin-PE
    BERT        (2018)      -> Encoder-only + MLM 预训练 (双向注意力)
    GPT-3       (2020)      -> Decoder-only + MHA + GELU-FFN + LN
    LLaMA       (2023)      -> GQA + SwiGLU + RMSNorm + RoPE (现代模板)
    Mixtral     (2024)      -> LLaMA + sparse MoE (softmax top-k)
    Mamba       (2023)      -> 非注意力 Selective SSM, O(T) 线性复杂度
    DeepSeek-V3 (2024)      -> MLA + 细粒度 MoE(SwiGLU) + RMSNorm + RoPE
    DeepSeek-V3.2 (2025)    -> DSA (Lightning Indexer + MLA) + MoE

多模态 (眼耳)
    CLIP        (2021)      -> 对比学习双塔, 视觉-语言对齐
    Whisper     (2022)      -> 音频 Encoder-Decoder, ASR 经典
    Qwen2-VL    (2024)      -> ViT + Resampler + M-RoPE 多模态 LLM
    Qwen2.5-Omni(2025)      -> Thinker-Talker 全模态 (文本+图像+音频+视频)

生成模型 (右脑)
    VAE / 3D VAE            -> Latent Diffusion 的前置压缩器
    DiT         (2023)      -> adaLN-Zero 取代 UNet, 成为扩散 Transformer 主干
    MM-DiT      (2024)      -> SD3/FLUX 双流同 attention, Rectified Flow 目标
    Video DiT   (2024)      -> Sora 风格 Spacetime Patches + DiT
    VAR         (2024)      -> 自回归图像生成, 复用 GPT 框架

使用示例:
    >>> from llm_models import GPT3, LLaMA, DeepSeekV3, DiT
    >>> from llm_models import Trainer, TrainingConfig, DDPMScheduler, DiffusionLoss
"""

__version__ = "0.4.0"
__author__ = "LLM Team"

# --- Layers (底层零件) ---
from llm_models.layers import (
    # Attention 家族
    ScaledDotProductAttention,
    SingleHeadSelfAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    MultiHeadLatentSparseAttention,
    # Position
    SinPositionalEncoding,
    RotaryPositionalEncoding,
    MultimodalRotaryEmbedding,
    apply_rotary_pos_emb,
    # FFN
    FeedForward,
    GeLUFeedForward,
    SwiGLUFeedForward,
    # Norm
    RMSNorm,
    # Block
    PreLNBlock,
    PreLNCrossBlock,
    # MoE
    MixtralMoE,
    # SSM
    SelectiveSSM,
    # Diffusion 条件注入
    AdaLNZeroBlock,
    FinalLayer,
    TimestepEmbedding,
    modulate,
    # VQ
    VectorQuantizer,
)

# --- Models ---
from llm_models.models import (
    # Transformer / BERT / GPT
    Transformer,
    EncoderLayer,
    DecoderLayer,
    BERT,
    BERTEmbeddings,
    GPT3,
    GPTBlock,
    # LLaMA / Mistral / MTP / Mixtral / Mamba
    LLaMA,
    LlamaBlock,
    Mistral,
    MistralBlock,
    MTPLLaMA,
    MTPModule,
    MTPLoss,
    Qwen3Next,
    Mixtral,
    MixtralBlock,
    Mamba,
    MambaBlock,
    MambaLayer,
    # DeepSeek
    DeepSeekV3,
    DeepSeekBlock,
    DeepSeekMoE,
    DeepSeekV3_2,
    DeepSeekV32Block,
    # Multimodal primitives
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
    # CLIP / Whisper
    CLIPModel,
    CLIPTextEncoder,
    CLIPVisionEncoder,
    Whisper,
    WhisperAudioEncoder,
    WhisperTextDecoder,
    # Qwen2-VL / Qwen2.5-Omni
    Qwen2VLDecoder,
    Qwen2VLModel,
    OmniThinkerDecoder,
    OmniTalkerDecoder,
    Qwen2_5_OmniModel,
    # Generation: VAE / DiT / MM-DiT / Video DiT / VAR
    ImageVAE,
    ImageVAEEncoder,
    ImageVAEDecoder,
    CausalVideoVAE,
    CausalVAE3DEncoder,
    CausalVAE3DDecoder,
    DiT,
    PatchifyConv,
    VideoDiT,
    Patchify3D,
    MMDiT,
    MMDiTBlock,
    ImageTokenizer,
    VARModel,
)

# --- Utils ---
from llm_models.utils import (
    get_pad_mask,
    get_subsequent_mask,
    build_causal_mask,
    build_sliding_window_mask,
    combine_causal_and_padding_mask,
    combine_masks,
)

# --- Training ---
from llm_models.training import (
    Trainer,
    TrainingConfig,
    # Loss
    LossComputer,
    StandardLMLoss,
    MoELMLoss,
    OmniLoss,
    MaskedLMLoss,
    ContrastiveLoss,
    VAELoss,
    VARLoss,
    DiffusionLoss,
    # Data
    SyntheticDataGenerator,
    DecoderOnlyDataGenerator,
    EncoderDecoderDataGenerator,
    VisionLanguageDataGenerator,
    OmniDataGenerator,
    MaskedLMDataGenerator,
    CLIPDataGenerator,
    WhisperDataGenerator,
    ImageDataGenerator,
    DiffusionDataGenerator,
    VideoDiffusionDataGenerator,
    VARImageDataGenerator,
    # Diffusion
    NoiseScheduler,
    DDPMScheduler,
    FlowMatchingScheduler,
    DDIMSampler,
    EulerFlowSampler,
    classifier_free_guidance,
)

__all__ = [
    # ---- Layers ----
    "ScaledDotProductAttention", "SingleHeadSelfAttention",
    "MultiHeadAttention", "GroupedQueryAttention",
    "MultiHeadLatentAttention", "MultiHeadLatentSparseAttention",
    "SinPositionalEncoding", "RotaryPositionalEncoding",
    "MultimodalRotaryEmbedding", "apply_rotary_pos_emb",
    "FeedForward", "GeLUFeedForward", "SwiGLUFeedForward",
    "RMSNorm", "PreLNBlock", "PreLNCrossBlock",
    "MixtralMoE", "SelectiveSSM",
    "AdaLNZeroBlock", "FinalLayer", "TimestepEmbedding", "modulate",
    "VectorQuantizer",
    # ---- Models: left-brain LLM ----
    "Transformer", "EncoderLayer", "DecoderLayer",
    "BERT", "BERTEmbeddings",
    "GPT3", "GPTBlock",
    "LLaMA", "LlamaBlock",
    "Mistral", "MistralBlock",
    "MTPLLaMA", "MTPModule", "MTPLoss",
    "Qwen3Next",
    "Mixtral", "MixtralBlock",
    "Mamba", "MambaBlock", "MambaLayer",
    "DeepSeekV3", "DeepSeekBlock", "DeepSeekMoE",
    "DeepSeekV3_2", "DeepSeekV32Block",
    # ---- Models: multimodal understanding ----
    "PatchEmbed2D", "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock", "PerceiverResampler",
    "ModalityProjector",
    "CLIPModel", "CLIPTextEncoder", "CLIPVisionEncoder",
    "Whisper", "WhisperAudioEncoder", "WhisperTextDecoder",
    "Qwen2VLDecoder", "Qwen2VLModel",
    "OmniThinkerDecoder", "OmniTalkerDecoder", "Qwen2_5_OmniModel",
    # ---- Models: right-brain generation ----
    "ImageVAE", "ImageVAEEncoder", "ImageVAEDecoder",
    "CausalVideoVAE", "CausalVAE3DEncoder", "CausalVAE3DDecoder",
    "DiT", "PatchifyConv",
    "VideoDiT", "Patchify3D",
    "MMDiT", "MMDiTBlock",
    "ImageTokenizer", "VARModel",
    # ---- Utils ----
    "get_pad_mask", "get_subsequent_mask", "build_causal_mask",
    "build_sliding_window_mask",
    "combine_causal_and_padding_mask", "combine_masks",
    # ---- Training ----
    "Trainer", "TrainingConfig",
    "LossComputer", "StandardLMLoss", "MoELMLoss", "OmniLoss",
    "MaskedLMLoss", "ContrastiveLoss", "VAELoss", "VARLoss", "DiffusionLoss",
    "SyntheticDataGenerator",
    "DecoderOnlyDataGenerator", "EncoderDecoderDataGenerator",
    "VisionLanguageDataGenerator", "OmniDataGenerator",
    "MaskedLMDataGenerator", "CLIPDataGenerator", "WhisperDataGenerator",
    "ImageDataGenerator", "DiffusionDataGenerator", "VideoDiffusionDataGenerator",
    "VARImageDataGenerator",
    "NoiseScheduler", "DDPMScheduler", "FlowMatchingScheduler",
    "DDIMSampler", "EulerFlowSampler", "classifier_free_guidance",
]
