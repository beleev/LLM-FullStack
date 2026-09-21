"""
损失函数模块
================

为不同模型架构封装对应的损失计算策略 (策略模式)。
Trainer 持有一个 LossComputer 实例，每步训练调用 `compute(model_output, labels)`，
对模型类型保持中立。

提供的策略:
    - StandardLMLoss : 标准下一 token 预测交叉熵 (GPT-3 / Transformer / LLaMA / Mamba / Whisper)
    - MTPLoss        : 主 CE + λ·多 token 预测 CE (MTPLLaMA / DeepSeek-V3 MTP)
    - MoELMLoss      : 交叉熵 + Switch-Transformer 风格的负载均衡 aux loss
                       (DeepSeekV3 / V3.2 / Mixtral)
    - OmniLoss       : 文本 (Thinker) + 音频 (Talker) 双分支加权 loss (Qwen2.5-Omni)
    - MaskedLMLoss   : BERT 风格 MLM 交叉熵 (只对被 mask 的位置算 loss)
    - ContrastiveLoss: CLIP 对称对比 loss (image↔text 双向 CE)
    - VAELoss        : 重建 (MSE) + KL(q || N(0, I))
    - VARLoss        : next-scale 交叉熵 (整级 token 并行预测) + 多尺度 VQ commitment
    - DiffusionLoss  : 见 training/diffusion.py

通用约定:
    - 标签使用 -100 作为 ignore_index (PyTorch cross_entropy 默认值)，
      pad / 多模态前缀 token 在该位置不参与梯度。
    - 返回 dict 必含 "total_loss" 字段，Trainer 调用其 `.backward()`；
      其余分量供日志监控，不直接反传。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class LossComputer(ABC):
    """
    损失计算基类 (策略模式接口)。

    子类必须实现 `compute()`，输入模型输出与标签，输出包含 "total_loss" 的 dict。
    """

    @abstractmethod
    def compute(
        self,
        model_output: Any,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失。

        Args:
            model_output: 模型 forward 的输出 (具体类型由子类约定)。
            labels:       目标标签。

        Returns:
            dict, 必须含 "total_loss"，可附加各分量 loss 用于日志。
        """
        raise NotImplementedError


class StandardLMLoss(LossComputer):
    """
    next-token 交叉熵: 所有 (batch, 位置) 的平均 NLL。未训练模型应 ≈ ln V (均匀瞎猜)。

    labels == -100 的位置被跳过 (PyTorch cross_entropy 的默认 ignore_index):
    数据侧只需把 pad / 多模态前缀位置填 -100。
    """

    def compute(
        self,
        model_output: torch.Tensor,     # logits [B, T, V]
        labels: torch.Tensor,           # [B, T]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits = model_output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, V]  cross_entropy 要 (N, C)
            labels.reshape(-1),                    # [B*T]
            ignore_index=-100,
        )
        # lm_loss 与 total_loss 此处相同，但保留两个键便于日志接口统一
        return {"total_loss": loss, "lm_loss": loss}


class MTPLoss(LossComputer):
    """
    Multi-Token Prediction 联合损失 (配 models/language_models/mtp.py::MTPLLaMA):

        L = CE(main) + λ · mean_k CE(mtp_k)          DeepSeek-V3: λ = 0.3 → 0.1

    标签对齐 (labels[i] = t_{i+1} 是标准 next-token 标签):
        MTP-k 在位置 i 预测 t_{i+1+k} = labels[i+k]
        → labels 左移 k 位作第 k 级目标, 末尾 k 个位置没有未来 token, 置 -100
    """

    def __init__(self, mtp_lambda: float = 0.3, ignore_index: int = -100) -> None:
        self.mtp_lambda = mtp_lambda
        self.ignore_index = ignore_index

    def _ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*T, V]
            labels.reshape(-1),                    # [B*T]
            ignore_index=self.ignore_index,
        )

    def compute(
        self,
        model_output: Dict[str, Any],   # {"logits": [B,T,V], "mtp_logits": List[[B,T,V]]}
        labels: torch.Tensor,           # [B, T]
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        main_loss = self._ce(model_output["logits"], labels)

        mtp_losses: List[torch.Tensor] = []
        for k, logits_k in enumerate(model_output["mtp_logits"], start=1):
            labels_k = torch.full_like(labels, self.ignore_index)
            labels_k[:, :-k] = labels[:, k:]       # 目标整体左移 k 位
            mtp_losses.append(self._ce(logits_k, labels_k))

        mtp_loss = torch.stack(mtp_losses).mean()
        return {
            "total_loss": main_loss + self.mtp_lambda * mtp_loss,
            "main_loss": main_loss.detach(),
            "mtp_loss": mtp_loss.detach(),
        }


class MoELMLoss(LossComputer):
    """
    MoE 语言模型损失 (Mixtral / DeepSeekV3 / V3.2):

        total = lm_loss + aux_loss_weight · aux_loss  [+ index_loss_weight · index_loss]

    aux_loss (Switch-Transformer 负载均衡), 每层:
        f_i = 专家 i 被选中的次数 / token 数        # 不可导; top-K 下 Σ_i f_i = K
        P_i = mean_t p_t[i]                         # 可导; p_t 是归一化到行和为 1 的路由概率
        aux = E · Σ_i f_i · P_i
      完全均衡时 f_i = K/E, P_i = 1/E → aux = E·E·(K/E)(1/E) = **K** (不是 1; top-1 时才是 1)。
      路由坍塌到固定 K 个专家时 → aux ≈ E。梯度只经 P_i 回到 router, 所以 router_logits 不能 detach。
      p_t 取自 routing_info["routing_probs"]: Mixtral 是 softmax (本来就归一),
      DeepSeek 是 sigmoid 分数, 这里除以行和 (与 V3 论文的 s'_i = s_i / Σ_j s_j 一致),
      而不是拿 router_logits 另算一遍与模型实际路由无关的 softmax。

    index_loss: DeepSeek-V3.2 的 indexer 对齐 KL, 由模型放在 routing_info["index_loss"];
      没有这个键的模型 (Mixtral / V3) 该项为 0, 不出现在返回 dict 里。

    Args:
        aux_loss_weight:   经验值 ~0.01; 太大牺牲主任务, 太小路由坍塌。
        index_loss_weight: indexer 与主模型参数不相交 (输入/目标都 detach), 取 1.0 即可。
    """

    def __init__(self, aux_loss_weight: float = 0.01, index_loss_weight: float = 1.0):
        self.aux_loss_weight = aux_loss_weight
        self.index_loss_weight = index_loss_weight

    def compute(
        self,
        model_output: tuple,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits, all_routing_info = model_output  # [B, T, V], 每层一个 routing_info dict

        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # 没有 MoE 层时返回与 logits 同设备的 0 (以前是 CPU 标量, GPU 上相加会报错)
        aux_loss = (
            self._compute_load_balancing_loss(all_routing_info)
            if all_routing_info else logits.new_zeros(())
        )
        out = {"lm_loss": lm_loss, "aux_loss": aux_loss}
        total_loss = lm_loss + self.aux_loss_weight * aux_loss

        index_losses = [info["index_loss"] for info in all_routing_info if "index_loss" in info]
        if index_losses:
            out["index_loss"] = torch.stack(index_losses).mean()
            total_loss = total_loss + self.index_loss_weight * out["index_loss"]

        return {"total_loss": total_loss, **out}

    @staticmethod
    def _compute_load_balancing_loss(
        all_routing_info: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """跨层平均的 E·Σ f_i·P_i; 完全均衡 = K, 完全坍塌 = E。all_routing_info 非空。"""
        layer_losses = []
        for info in all_routing_info:
            probs = info["routing_probs"].float()              # [N, E] float32 防 fp16 溢出
            probs = probs / probs.sum(dim=-1, keepdim=True)    # sigmoid 分数 → 行和为 1
            num_experts = probs.size(-1)

            # f_i: one-hot 计数后按 token 求均值 (统计量, 无梯度)
            fraction = F.one_hot(info["selected_experts"], num_experts).sum(dim=1).float().mean(dim=0)  # [E]
            mean_prob = probs.mean(dim=0)                                                               # [E]
            layer_losses.append(num_experts * (fraction * mean_prob).sum())

        return torch.stack(layer_losses).mean()


class OmniLoss(LossComputer):
    """
    全模态 (Thinker + Talker) 双分支损失。

    适用模型: Qwen2.5-Omni

    总损失:
        total_loss = text_loss + audio_loss_weight * audio_loss

    - text_loss : Thinker (主 LLM) 的 next-token 交叉熵, 监督文本生成；
    - audio_loss: Talker (语音头) 对离散音频 token 的自回归交叉熵, 可选。

    为什么音频 loss 可选？
        训练数据可能只有文本标注 (无音频 ground truth)；此时 audio_logits / labels
        缺失，本类自动退化为纯文本 loss，避免硬报错。

    audio_loss_weight 的作用:
        平衡两条监督信号的强度。文本 loss 通常更稳定且收敛慢, 音频 loss 数值范围
        可能不同；可调系数让两个分支共同进步而不互相压制。

    Args:
        audio_loss_weight: 音频 loss 在总损失中的相对权重。
    """

    def __init__(self, audio_loss_weight: float = 0.5):
        self.audio_loss_weight = audio_loss_weight

    def compute(
        self,
        model_output: Dict[str, Optional[torch.Tensor]],
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        text_logits = model_output["text_logits"]
        audio_logits = model_output.get("audio_logits")  # 可能为 None

        # 1) 文本 loss: 与标准 LM 相同, 多模态前缀位置已通过 -100 屏蔽
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        result: Dict[str, torch.Tensor] = {
            "text_loss": text_loss,
        }

        # 2) 音频 loss (可选): 只有当 Talker 有输出且 batch 提供 audio_labels 时才计算
        audio_labels = kwargs.get("audio_labels")
        if audio_logits is not None and audio_labels is not None:
            audio_loss = F.cross_entropy(
                audio_logits.reshape(-1, audio_logits.size(-1)),
                audio_labels.reshape(-1),
                ignore_index=-100,
            )
            result["audio_loss"] = audio_loss
            result["total_loss"] = text_loss + self.audio_loss_weight * audio_loss
        else:
            # 退化为纯文本损失
            result["total_loss"] = text_loss

        return result


class MaskedLMLoss(LossComputer):
    """
    BERT 风格 Masked Language Modeling 损失。

    适用模型: BERT

    与 StandardLMLoss 的差异:
        - 只对 **被 mask 的位置** 算 loss (labels 其余位置填 -100)
        - 输入输出形状与 standard LM 相同 (都是 [B, T, V])
        - 语义不同: BERT 是"重建被遮盖的 token", 不是"预测下一个"

    训练数据构造:
        在 BertMLMDataGenerator 中, 随机挑 15% 位置做 mask, 其中:
            80% 换成 [MASK] token id
            10% 换成随机 token
            10% 保持原样
        对应位置的 label 设为原 token id, 其他位置填 -100。
    """

    def compute(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits = model_output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return {"total_loss": loss, "mlm_loss": loss}


class ContrastiveLoss(LossComputer):
    """
    CLIP 对称对比 loss (InfoNCE 形式)

    给定 batch 内 B 对 (image, text):
        logits_per_image = logit_scale · image_feats @ text_feats^T        # [B, B]
        logits_per_text  = logits_per_image.T
        对角线为正样本, 其余为负样本
        loss = (CE(logits_per_image, arange(B)) + CE(logits_per_text, arange(B))) / 2

    为什么要 "对称 CE"?
        单向 CE (只做 image→text 检索) 会让温度不对称地压缩其中一侧;
        取两向平均让模型在两侧都保持判别力。

    model_output 必须是 CLIPModel.forward 的返回 dict:
        image_features: [B, D]  (已 L2 normalize)
        text_features:  [B, D]  (已 L2 normalize)
        logit_scale:    scalar
    """

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: Any = None,  # 不需要显式 labels (对角线隐式给出)
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        image_feats = model_output["image_features"]
        text_feats = model_output["text_features"]
        logit_scale = model_output["logit_scale"]

        B = image_feats.size(0)
        logits = logit_scale * image_feats @ text_feats.t()     # [B, B]
        targets = torch.arange(B, device=logits.device)

        loss_i2t = F.cross_entropy(logits, targets)             # image → text
        loss_t2i = F.cross_entropy(logits.t(), targets)         # text  → image
        loss = (loss_i2t + loss_t2i) / 2

        return {
            "total_loss": loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "logit_scale": logit_scale.detach(),
        }


class VAELoss(LossComputer):
    """
    VAE 重建 + KL 正则

    loss = recon_weight · MSE(x̂, x) + kl_weight · KL(q || N(0, I))
    KL 闭式: -0.5 · Σ (1 + logσ² - μ² - σ²)

    适用模型: ImageVAE, CausalVideoVAE (只要 forward 返回 {recon, mean, logvar}
    且 batch 中的 "labels" 实为原输入 x)

    Args:
        recon_weight: 重建项系数 (默认 1.0)
        kl_weight:    KL 项系数 (默认 1e-4, SD 1.5 用到 1e-6 级, 控制潜空间"紧致度")
    """

    def __init__(self, recon_weight: float = 1.0, kl_weight: float = 1e-4):
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        recon = model_output["recon"]
        mean = model_output["mean"]
        logvar = model_output["logvar"]

        recon_loss = F.mse_loss(recon, labels)
        # KL(q(z|x) || N(0, I)) 逐样本求和, 再平均
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # 先对 latent 维度求和, 再对 batch 取均值, 让 KL 与模型规模无关
        kl_loss = kl.flatten(1).sum(dim=1).mean()

        total = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class VARLoss(LossComputer):
    """
    VAR (next-scale prediction) loss: 对 token 金字塔全部 L=Σs² 个位置做交叉熵。

    tokenizer 已冻结, 只训 Transformer。没有 shift-by-one —— 第 k 级位置的输入来自
    更粗的级, label 是本级 token, 对齐由 VARModel.forward 完成:
        logits: [B, L, K]   labels: [B, L]   (K = 码本大小, 初始 CE ≈ ln K)

    Trainer 传入的 labels 参数被忽略 (token 要经 tokenizer 才有, 由 model_output 自带)。
    """

    def compute(
        self,
        model_output: Dict[str, torch.Tensor],
        labels: Any = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        logits = model_output["logits"]
        target = model_output["labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )
        return {"total_loss": loss, "ce_loss": loss}
