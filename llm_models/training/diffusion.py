"""
扩散 / 流匹配 调度器 + 采样器 + CFG 工具

本文件为 Image / Video DiT 与 MM-DiT 提供训练 & 采样所需的全部支持:

1) 噪声调度器 (Scheduler):
    - DDPMScheduler           : 经典 ε-prediction, cosine / linear β 调度
    - FlowMatchingScheduler   : Rectified Flow / SD3 路线, velocity-prediction
2) 采样器 (Sampler, 推理时反向去噪):
    - DDIMSampler             : Denoising Diffusion Implicit (少步确定性采样)
    - EulerFlowSampler        : Rectified Flow 的 Euler ODE 求解
3) Loss:
    - DiffusionLoss           : 自动按 scheduler 的 target 类型 (ε 或 v) 算 MSE
4) CFG 辅助:
    - classifier_free_guidance: 给定 "有条件" 与 "无条件" 两套 pred, 做线性外插

设计要点:
    - Scheduler 负责 "正向加噪 q(x_t | x_0)" 与 "损失目标 pred 的语义"
    - Sampler   负责 "反向去噪" (从 x_T 走到 x_0)
    - 二者解耦, 可自由组合; 这也是 HuggingFace diffusers 库的风格
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_models.training.loss import LossComputer


# -----------------------------------------------------------------------------
# Schedulers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AddNoiseResult:
    """
    正向加噪 q(x_t | x_0) 的输出包:
        noisy:  x_t = forward(x_0, ε, t)
        noise:  ε (训练 ε-pred 时作为 target)
        target: 真正用作 loss target 的量
                - DDPM (ε-pred) 下 = noise
                - Flow Matching (v-pred) 下 = velocity
        t_norm: 与 model.forward 一致的时间步张量 (已归一化到 [0, 1] 或保持原始步数)
    """
    noisy: torch.Tensor
    noise: torch.Tensor
    target: torch.Tensor
    t_norm: torch.Tensor


class NoiseScheduler(ABC):
    """
    抽象调度器: 定义 forward (加噪) + 训练 target + 推理 step 的语义。
    """

    prediction_type: str  # "epsilon" | "velocity"

    @abstractmethod
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从调度器定义的分布中采训练时的 timestep。"""

    @abstractmethod
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> AddNoiseResult:
        """正向加噪 q(x_t | x_0), 产出 loss target。"""


class DDPMScheduler(NoiseScheduler):
    """
    DDPM (Ho et al., 2020) cosine β 调度 + ε-prediction

    加噪: x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε
    target = ε

    为什么用 cosine 调度 (Nichol & Dhariwal, 2021)?
        linear β 在 T 较大时端点噪声过大, cosine 调度让 ᾱ_t 更平滑, 生成质量更好。
    """

    prediction_type = "epsilon"

    def __init__(self, num_train_timesteps: int = 1000, s: float = 0.008):
        self.num_train_timesteps = num_train_timesteps

        # cosine ᾱ_t = f(t)^2 / f(0)^2, f(t) = cos((t/T + s)/(1+s) · π/2)
        t = torch.arange(num_train_timesteps + 1, dtype=torch.float) / num_train_timesteps
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(max=0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # DDPM 训练时均匀采 t ∈ [0, T-1]
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    def _broadcast(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """把 [B] 张量扩到 ref 的维度 (广播到 [B, 1, 1, ...]) 以便和空间张量相乘。"""
        while x.dim() < ref.dim():
            x = x.unsqueeze(-1)
        return x

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> AddNoiseResult:
        noise = torch.randn_like(x0)
        sqrt_ab = self._broadcast(self.sqrt_alphas_cumprod[t], x0)
        sqrt_1mab = self._broadcast(self.sqrt_one_minus_alphas_cumprod[t], x0)
        noisy = sqrt_ab * x0 + sqrt_1mab * noise
        # DDPM 传给模型的 t 直接用原始离散步数 (float 化, TimestepEmbedding 可吃连续值)
        return AddNoiseResult(noisy=noisy, noise=noise, target=noise, t_norm=t.float())


class FlowMatchingScheduler(NoiseScheduler):
    """
    Rectified Flow / Flow Matching (SD3, FLUX)

    线性路径: x_t = (1 - t) · x_0 + t · ε,   t ∈ [0, 1]
    target = velocity = dx_t/dt = ε - x_0

    一句话差异对比:
        DDPM: 学噪声 ε, 用 cosine 调度加噪, 推理数十步
        Flow Matching: 学 velocity v, 用线性直线路径, 推理数步即可 (轨迹更直)

    训练时 t 的分布:
        原论文用 logit-normal (偏重中段 t), 教学用均匀分布, 实现更简单。
    """

    prediction_type = "velocity"

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps

    def to(self, device: torch.device):
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # 直接采 [0, 1] 的连续值, 比 DDPM 的离散 t 更自然
        return torch.rand(batch_size, device=device)

    @staticmethod
    def _broadcast(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        while x.dim() < ref.dim():
            x = x.unsqueeze(-1)
        return x

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> AddNoiseResult:
        noise = torch.randn_like(x0)
        t_b = self._broadcast(t, x0)
        noisy = (1 - t_b) * x0 + t_b * noise
        velocity = noise - x0  # dx_t/dt, 与 t 无关 (直线路径)
        return AddNoiseResult(noisy=noisy, noise=noise, target=velocity, t_norm=t)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


class DiffusionLoss(LossComputer):
    """
    按 scheduler 的 prediction_type 计算 MSE loss。

    训练约定 (与 Trainer 的接口对齐):
        model_output: 模型输出的 pred (形状与 target 相同, e.g. [B, C, H, W])
        labels:       scheduler 产生的 target (noise 或 velocity)
        kwargs:       可传 "loss_mask" (同形 mask), 用于把部分位置排除

    为什么要独立的 DiffusionLoss?
        Trainer 默认调 LossComputer.compute(out, labels), 扩散训练的 target 不是
        "下一 token 分类", 而是 "MSE 回归". 在 training/data.py 的扩散生成器里,
        我们已经让 labels = scheduler.target, 所以此处只做 MSE 即可。
    """

    def compute(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        loss = F.mse_loss(model_output, labels)
        return {"total_loss": loss, "diffusion_loss": loss}


# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------


class DDIMSampler:
    """
    DDIM (Song et al., 2021) 确定性采样器 — DDPM 的少步推理版

    从 T-1 走到 0, 每步根据模型预测的 ε 做:
        x_0 = (x_t - sqrt(1-ᾱ_t) · ε) / sqrt(ᾱ_t)
        x_{t-1} = sqrt(ᾱ_{t-1}) · x_0 + sqrt(1 - ᾱ_{t-1}) · ε

    若希望随机采样, 可在每步加入 noise (传统 DDPM 的 q(x_{t-1} | x_t, x_0));
    本实现保持纯确定性 (η=0)。

    用法:
        sampler = DDIMSampler(scheduler, num_inference_steps=50)
        x0 = sampler.sample(model, shape=(B, C, H, W), device=...)
    """

    def __init__(self, scheduler: DDPMScheduler, num_inference_steps: int = 50):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        null_class_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.scheduler.to(device)
        x = torch.randn(shape, device=device)

        # 均匀取子步, 覆盖 [0, T-1]
        step_ids = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0,
            self.num_inference_steps, device=device,
        ).long()

        for i in range(self.num_inference_steps):
            t = step_ids[i]
            t_batch = t.expand(shape[0])

            pred = _apply_cfg(model, x, t_batch, class_labels, guidance_scale, null_class_id)

            # DDIM 更新公式
            ab_t = self.scheduler.alphas_cumprod[t]
            if i < self.num_inference_steps - 1:
                ab_prev = self.scheduler.alphas_cumprod[step_ids[i + 1]]
            else:
                ab_prev = torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - ab_t).sqrt() * pred) / ab_t.sqrt()
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * pred

        return x


class EulerFlowSampler:
    """
    Rectified Flow 的 Euler ODE 求解 — 线性路径, 步长 Δt = 1/N

    从 x_1 = ε (纯噪声) 开始, 每步:
        v = model(x_t, t)
        x_{t-Δt} = x_t - Δt · v
    直到 t → 0, 即得 x_0。

    少步即可收敛 (教学默认 20 步; SD3 推理 28 步左右).
    """

    def __init__(self, num_inference_steps: int = 20):
        self.num_inference_steps = num_inference_steps

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        null_class_id: Optional[int] = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        # t 从 1 线性递减到 0
        ts = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)
        for i in range(self.num_inference_steps):
            t = ts[i]
            dt = ts[i] - ts[i + 1]
            t_batch = t.expand(shape[0])

            v = _apply_cfg(model, x, t_batch, class_labels, guidance_scale, null_class_id)
            x = x - dt * v
        return x


# -----------------------------------------------------------------------------
# Classifier-Free Guidance (CFG)
# -----------------------------------------------------------------------------


def classifier_free_guidance(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    CFG 线性外插:
        pred = uncond + guidance_scale · (cond - uncond)

    guidance_scale = 1  → 只用条件 (不引导)
    guidance_scale > 1  → 推模型更靠近条件, 典型 7.5 (SD 系列默认)
    """
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)


def _apply_cfg(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    class_labels: Optional[torch.Tensor],
    guidance_scale: float,
    null_class_id: Optional[int],
) -> torch.Tensor:
    """
    sampler 内部用的 CFG 封装:
        - class_labels 为 None 或 guidance_scale == 1 时直接单次前向
        - 否则额外跑一次 null class, 线性外插
    """
    if class_labels is None or guidance_scale == 1.0:
        return model(x, t, class_labels)

    if null_class_id is None:
        raise ValueError("CFG 需要 null_class_id (训练时用 class_dropout 制造的 null)")

    uncond_labels = torch.full_like(class_labels, null_class_id)
    cond_pred = model(x, t, class_labels)
    uncond_pred = model(x, t, uncond_labels)
    return classifier_free_guidance(cond_pred, uncond_pred, guidance_scale)
