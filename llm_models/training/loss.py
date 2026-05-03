"""
损失函数模块
================

为不同模型架构封装对应的损失计算策略 (策略模式)。
Trainer 持有一个 LossComputer 实例，每步训练调用 `compute(model_output, labels)`，
对模型类型保持中立。

提供的策略:
    - StandardLMLoss : 标准下一 token 预测交叉熵 (GPT-3 / Transformer / LLaMA / Mamba / Whisper)
    - MoELMLoss      : 交叉熵 + Switch-Transformer 风格的负载均衡 aux loss
                       (DeepSeekV3 / V3.2 / Mixtral)
    - OmniLoss       : 文本 (Thinker) + 音频 (Talker) 双分支加权 loss (Qwen2.5-Omni)
    - MaskedLMLoss   : BERT 风格 MLM 交叉熵 (只对被 mask 的位置算 loss)
    - ContrastiveLoss: CLIP 对称对比 loss (image↔text 双向 CE)
    - VAELoss        : 重建 (MSE) + KL(q || N(0, I))
    - VARLoss        : next-token 交叉熵 + VQ-VAE commitment (若在联合训练)
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
    标准语言模型 (next-token prediction) 损失。

    适用模型: GPT-3, 经典 Transformer (decoder 端)。

    实现:
        loss = cross_entropy(
            logits.view(-1, V),       # 把 (B, T, V) 展平为 (B*T, V)
            labels.view(-1),          # (B*T,)
            ignore_index=-100,        # 跳过 pad 位置, 避免污染 loss
        )

    为什么要 reshape？
        nn.functional.cross_entropy 期望输入维度为 (N, C)；展平 batch 与时间维
        即可一次性算出所有 token 的平均 NLL，避免 Python for-loop。

    为什么用 -100 作为 ignore？
        PyTorch 的 cross_entropy 默认就把 target=-100 跳过。约定俗成，
        让数据生成端只需把 pad / 模态前缀位置填 -100 即可。
    """

    def compute(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # model_output: logits, 形状 [B, T, V]
        logits = model_output
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        # lm_loss 与 total_loss 此处相同，但保留两个键便于日志接口统一
        return {"total_loss": loss, "lm_loss": loss}


class MoELMLoss(LossComputer):
    """
    MoE (Mixture-of-Experts) 语言模型损失。

    适用模型: DeepSeekV3, DeepSeekV3_2

    总损失:
        total_loss = lm_loss + aux_loss_weight * aux_loss

    为什么需要 aux loss (负载均衡)？
        MoE 通过 router 把每个 token 分发给少量专家 (top-k)。若不加约束，
        router 会迅速 "坍塌"——只把 token 送给少数明星专家，其余专家拿不到
        梯度永远不被训练。这既浪费参数容量，又损害模型表达力。
        Switch Transformer 提出的辅助 loss 鼓励 token 在专家间均匀分布。

    为什么 router_logits 不能 detach？
        aux loss 的目的就是更新 router 参数；若 detach，梯度无法回传到
        router 的线性层，aux loss 等于白算。同时 router 也通过 top-k 加权
        的 expert 输出从主 LM loss 拿到梯度，两者协同。

    aux_loss_weight 的取值权衡:
        - 太大: 模型为 "均衡" 牺牲表达力，主任务退化；
        - 太小: 路由仍会坍塌；
        - 经验: ~0.01 (DeepSeek / Switch Transformer 常用)。

    Args:
        aux_loss_weight: 辅助 loss 在总损失中的权重。
    """

    def __init__(self, aux_loss_weight: float = 0.01):
        self.aux_loss_weight = aux_loss_weight

    def compute(
        self,
        model_output: tuple,
        labels: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 模型输出: (logits, all_routing_info)
        # all_routing_info 是每个 MoE 层的 router 中间量列表
        logits, all_routing_info = model_output

        # 1) 主任务: 标准 next-token 交叉熵
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # 2) 辅助任务: 鼓励 token 在专家间均匀分配
        aux_loss = self._compute_load_balancing_loss(all_routing_info)

        total_loss = lm_loss + self.aux_loss_weight * aux_loss
        return {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
        }

    def _compute_load_balancing_loss(
        self, all_routing_info: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        计算 Switch-Transformer 风格负载均衡 loss。

        定义:
            f_i = (被路由到专家 i 的 token 数) / (token 总数)        # 离散统计量
            P_i = mean_t softmax(router_logits[t])[i]               # 平均路由概率
            aux = N * Σ_i (f_i * P_i),  N 为专家总数

        直觉:
            - f_i 反映 "实际分配比例" (但来自 argmax/top-k, 不可微)；
            - P_i 反映 "router 的偏好"        (来自 softmax, 可微)；
            - 二者相乘并按专家求和，再乘 N：完美均匀分配时 f_i = P_i = 1/N，
              和为 N * N * (1/N)^2 = 1，即理想下界为 1；
            - 任何不均衡都会让该乘积之和增大，从而推动 router 学习更均衡的偏好。

        梯度路径:
            f_i 不可微 (含 argmax/scatter)，但 P_i 可微，最终梯度通过 P_i
            反传更新 router 权重。
        """
        # 模型可能未启用 MoE / 没有 routing 记录, 安全返回 0
        if not all_routing_info:
            return torch.tensor(0.0)

        # 累加各 MoE 层的 aux loss, 最后取均值
        total_aux = torch.tensor(0.0, device=all_routing_info[0]["router_logits"].device)

        for info in all_routing_info:
            router_logits = info["router_logits"]      # [total_tokens, num_experts]
            selected_experts = info["selected_experts"]  # [total_tokens, top_k]
            num_experts = router_logits.size(-1)
            num_tokens = router_logits.size(0)

            # P_i: 各专家平均路由概率 (可微分量, 梯度从这里回传)
            # 升 float32 计算 softmax, 防止 fp16 溢出导致 NaN
            routing_probs = F.softmax(router_logits.float(), dim=-1)  # [T, E]
            mean_prob = routing_probs.mean(dim=0)                     # [E]

            # f_i: 被选中的 token 比例 (统计量, 不参与梯度)
            # 用 scatter_ 把 selected_experts 索引位置置 1，再按列求均值
            expert_mask = torch.zeros(
                num_tokens, num_experts,
                device=router_logits.device, dtype=routing_probs.dtype,
            )
            expert_mask.scatter_(1, selected_experts, 1.0)
            fraction = expert_mask.mean(dim=0)  # [E]

            # 单层 aux loss
            layer_aux = num_experts * (fraction * mean_prob).sum()
            total_aux = total_aux + layer_aux

        # 跨层取均值, 让 aux_loss_weight 的物理含义不随层数变化
        return total_aux / len(all_routing_info)


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
    VAR (next-token autoregressive image) loss

    VAR 的 tokenizer 通常单独预训练并冻结, 此处只对 GPT 分支算 next-token 交叉熵。

    model_output 是 VARModel.forward 返回的 dict:
        logits: [B, N, vocab]
        labels: [B, N]

    Trainer 接口: 传入的 labels 参数可被忽略 (由 model_output 自带)。
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
