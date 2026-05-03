"""
multimodal — 跨模态构建块

  PatchEmbed2D / PatchEmbed3D: 图像 / 视频 patch 化
  PatchTransformerEncoder:    标准 ViT 风格编码器
  PerceiverResampler:         可变长 token 重采样 (Flamingo 系)
  ModalityProjector:          模态间投影桥
"""

from llm_models.layers.multimodal.multimodal import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchTransformerEncoder,
    PerceiverResamplerBlock,
    PerceiverResampler,
    ModalityProjector,
)

__all__ = [
    "PatchEmbed2D",
    "PatchEmbed3D",
    "PatchTransformerEncoder",
    "PerceiverResamplerBlock",
    "PerceiverResampler",
    "ModalityProjector",
]
