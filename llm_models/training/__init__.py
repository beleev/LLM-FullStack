"""
训练模块包导出
================

本子包提供一套 "策略模式" 风格的通用训练框架, 适配本教学库的全部模型,
从 LM (GPT/LLaMA/Mamba) 到 MoE (Mixtral/DeepSeek) 到多模态 (CLIP/Whisper/Omni)
到生成模型 (VAE/DiT/MM-DiT/Video DiT/VAR)。

核心组件:
    - Trainer               : 训练循环容器, 与具体模型/损失/数据无关
    - TrainingConfig        : 不可变 (frozen) 训练配置
    - LossComputer 子类     : 为不同模型架构封装损失
    - SyntheticDataGenerator 子类 : 为不同模型架构提供合成 batch

扩散专用:
    - DDPMScheduler / FlowMatchingScheduler : 噪声调度
    - DDIMSampler  / EulerFlowSampler       : 采样
    - DiffusionLoss                          : MSE loss (自动按 scheduler 的 target 类型)
    - classifier_free_guidance               : CFG 线性外插工具

典型用法:
    >>> cfg = TrainingConfig()
    >>> data_gen = DecoderOnlyDataGenerator(...)
    >>> loss_fn = StandardLMLoss()
    >>> trainer = Trainer(model, cfg, data_gen, loss_fn)
    >>> trainer.train()
"""

from llm_models.training.config import TrainingConfig
from llm_models.training.trainer import Trainer
from llm_models.training.loss import (
    LossComputer,
    StandardLMLoss,
    MoELMLoss,
    OmniLoss,
    MaskedLMLoss,
    ContrastiveLoss,
    VAELoss,
    VARLoss,
)
from llm_models.training.data import (
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
)
from llm_models.training.diffusion import (
    NoiseScheduler,
    DDPMScheduler,
    FlowMatchingScheduler,
    DDIMSampler,
    EulerFlowSampler,
    DiffusionLoss,
    classifier_free_guidance,
)

__all__ = [
    # 配置
    "TrainingConfig",
    # 训练循环
    "Trainer",
    # 损失策略
    "LossComputer",
    "StandardLMLoss",
    "MoELMLoss",
    "OmniLoss",
    "MaskedLMLoss",
    "ContrastiveLoss",
    "VAELoss",
    "VARLoss",
    "DiffusionLoss",
    # 数据生成策略
    "SyntheticDataGenerator",
    "DecoderOnlyDataGenerator",
    "EncoderDecoderDataGenerator",
    "VisionLanguageDataGenerator",
    "OmniDataGenerator",
    "MaskedLMDataGenerator",
    "CLIPDataGenerator",
    "WhisperDataGenerator",
    "ImageDataGenerator",
    "DiffusionDataGenerator",
    "VideoDiffusionDataGenerator",
    "VARImageDataGenerator",
    # 扩散调度 & 采样
    "NoiseScheduler",
    "DDPMScheduler",
    "FlowMatchingScheduler",
    "DDIMSampler",
    "EulerFlowSampler",
    "classifier_free_guidance",
]
