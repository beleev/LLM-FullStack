"""
模型模块

按架构/用途分为四大类:

语言模型 (Language Models)
- Transformer: Encoder-Decoder (Vaswani et al., 2017)
- BERT: Encoder-only, MLM (Devlin et al., 2018)
- GPT-3: Decoder-only, 自回归 LM (Brown et al., 2020)
- LLaMA: 现代 Decoder-only (Touvron et al., 2023)
- Mamba: Selective SSM, O(T) (Gu & Dao, 2023)

MoE 模型 (Mixture of Experts)
- Mixtral: Sparse MoE (Jiang et al., 2024)
- DeepSeek-V3 / V3.2: MLA + MoE + DSA (2024-2025)

多模态模型 (Multimodal)
- CLIP: 图文对比学习 (Radford et al., 2021)
- Whisper: 语音识别 (Radford et al., 2022)
- Qwen2-VL / Qwen2.5-Omni: 多模态 LLM (2024-2025)
- 公共构件: PatchEmbed, Perceiver, ModalityProjector

生成模型 (Generative)
- VAE / 3D VAE: Latent 压缩器
- DiT / MM-DiT / Video DiT: 扩散 Transformer
- VAR: 自回归图像生成
"""

# --- 基础组件 ---
from llm_models.models.foundation.transformer import (
    Transformer,
    EncoderLayer,
    DecoderLayer,
)

# --- 语言模型 ---
from llm_models.models.language_models.bert import BERT, BERTEmbeddings
from llm_models.models.language_models.gpt3 import GPT3, GPTBlock
from llm_models.models.language_models.llama import LLaMA, LlamaBlock
from llm_models.models.language_models.mamba import Mamba, MambaBlock, MambaLayer

# --- MoE ---
from llm_models.models.moe.mixtral import Mixtral, MixtralBlock
from llm_models.models.moe.deepseekV3 import (
    DeepSeekV3,
    DeepSeekBlock,
    DeepSeekMoE,
    DeepSeekV3_2,
    DeepSeekV32Block,
)

# --- 多模态 ---
from llm_models.layers.multimodal import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
)
from llm_models.models.multimodal.clip import (
    CLIPModel,
    CLIPTextEncoder,
    CLIPVisionEncoder,
)
from llm_models.models.multimodal.whisper import (
    Whisper,
    WhisperAudioEncoder,
    WhisperTextDecoder,
)
from llm_models.models.multimodal.qwen2_vl import Qwen2VLDecoder, Qwen2VLModel
from llm_models.models.multimodal.qwen2_5_omni import (
    OmniThinkerDecoder,
    OmniTalkerDecoder,
    Qwen2_5_OmniModel,
)

# --- 生成模型 ---
from llm_models.models.generative.vae import (
    ImageVAE,
    ImageVAEEncoder,
    ImageVAEDecoder,
)
from llm_models.models.generative.vae3d import (
    CausalVideoVAE,
    CausalVAE3DEncoder,
    CausalVAE3DDecoder,
)
from llm_models.models.generative.dit import DiT, PatchifyConv
from llm_models.models.generative.video_dit import VideoDiT, Patchify3D
from llm_models.models.generative.mmdit import MMDiT, MMDiTBlock
from llm_models.models.generative.var import ImageTokenizer, VARModel

__all__ = [
    # Language Models
    "Transformer", "EncoderLayer", "DecoderLayer",
    "BERT", "BERTEmbeddings",
    "GPT3", "GPTBlock",
    "LLaMA", "LlamaBlock",
    "Mamba", "MambaBlock", "MambaLayer",
    # MoE
    "Mixtral", "MixtralBlock",
    "DeepSeekV3", "DeepSeekBlock", "DeepSeekMoE",
    "DeepSeekV3_2", "DeepSeekV32Block",
    # Multimodal
    "PatchEmbed2D", "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock", "PerceiverResampler",
    "ModalityProjector",
    "CLIPModel", "CLIPTextEncoder", "CLIPVisionEncoder",
    "Whisper", "WhisperAudioEncoder", "WhisperTextDecoder",
    "Qwen2VLDecoder", "Qwen2VLModel",
    "OmniThinkerDecoder", "OmniTalkerDecoder", "Qwen2_5_OmniModel",
    # Generative
    "ImageVAE", "ImageVAEEncoder", "ImageVAEDecoder",
    "CausalVideoVAE", "CausalVAE3DEncoder", "CausalVAE3DDecoder",
    "DiT", "PatchifyConv",
    "VideoDiT", "Patchify3D",
    "MMDiT", "MMDiTBlock",
    "ImageTokenizer", "VARModel",
]
