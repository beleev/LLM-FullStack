"""
通用训练器模块
================

提供 Trainer 类, 支持本教学库内全部 LLM 模型的训练循环:
    - 标准 Decoder-only (GPT-3)
    - Encoder-Decoder   (经典 Transformer)
    - MoE              (DeepSeek-V3 / V3.2)
    - 多模态           (Qwen2-VL, Qwen2.5-Omni)

设计核心 - 策略模式:
    Trainer 不直接关心 "数据怎么来" 或 "损失怎么算", 而是依赖两个抽象:
        - SyntheticDataGenerator : 负责产出 batch
        - LossComputer           : 负责把模型输出转化为 loss
    任何新模型只需实现对应的 generator / loss, 即可复用同一套训练循环,
    保持训练器的开闭原则 (对扩展开放, 对修改关闭)。

性能小细节:
    - train_step 返回 tensor 形式的 metrics, 仅在日志打印时才 .item();
      避免每步都触发 GPU→CPU 同步, 在大模型训练中可显著减少阻塞。
"""

from typing import Dict, List, Union

import torch
import torch.nn as nn

from llm_models.training.config import TrainingConfig
from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer

# Metric: 训练步内部的指标值。tensor 用于 GPU 上累积, float 用于纯标量 (如 lr)。
Metric = Union[torch.Tensor, float]


class Trainer:
    """
    通用训练器。

    单步训练流程 (train_step):
        1) data_generator.generate_batch() → 取一个 batch
        2) 弹出 labels / 额外标签, 余下字段作为 model 的 forward 关键字参数
        3) 模型前向
        4) loss_computer.compute() → 计算总 loss 与各分量
        5) loss.backward()  反向传播
        6) clip_grad_norm_   梯度裁剪 (防爆炸)
        7) optimizer.step() / scheduler.step() / zero_grad()

    优化器选择: AdamW
        - Adam 对 Transformer 友好, 自适应学习率减轻调参负担;
        - W (Decoupled Weight Decay) 把权重衰减从梯度路径中解耦, 避免与 Adam
          的二阶动量耦合产生 "等效正则减弱" 问题, 是当今 LLM 训练的事实标准。
        - 注: 严格起见 norm/bias 等参数不应施加 weight decay
          (它们本就是小量, 正则会损害表达力), 教学版为简洁统一处理。

    学习率调度: LambdaLR + cosine warmup
        见 TrainingConfig.get_lr_lambda 文档。

    梯度裁剪: clip_grad_norm_
        Transformer 训练初期容易出现梯度尖峰 (注意力 softmax 饱和、初始化不佳等),
        一次大梯度可能直接破坏后续训练。裁剪到固定 L2 范数, 牺牲一点更新精度
        换取训练稳定。

    Args:
        model:          PyTorch 模型 (nn.Module)。
        config:         TrainingConfig。
        data_generator: 合成 batch 生成器 (策略)。
        loss_computer:  loss 计算器 (策略)。
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        data_generator: SyntheticDataGenerator,
        loss_computer: LossComputer,
    ):
        self.model = model
        self.config = config
        self.data_generator = data_generator
        self.loss_computer = loss_computer

        # AdamW: LLM 训练事实标准。weight_decay 解耦, 不被 Adam 的自适应缩放污染。
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Cosine warmup 调度: 先线性 warm up 到峰值再余弦退火至 0
        # 用 LambdaLR 把 lr 缩放函数挂到 base_lr 上, 自由度大且实现简洁
        lr_lambda = config.get_lr_lambda(config.num_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        self._global_step = 0

    def train_step(self) -> Dict[str, Metric]:
        """
        执行一步训练, 返回本步的训练指标。

        Returns:
            dict, 含各项 loss 张量、grad_norm 张量, 以及 lr (float)。

            为什么不直接转 float？
              .item() 会触发 device→host 同步, 在大模型 + 大 batch 下会阻塞 GPU
              流水线。这里保留 tensor, 仅在 `_log_metrics` 真正要打印时才转 float
              (而打印只在 log_interval 间隔发生)。
        """
        self.model.train()

        # ---- 1) 取数据 ----
        batch = self.data_generator.generate_batch()

        # ---- 2) 拆出 labels / extra_labels, 让 batch 中只剩 model.forward 入参 ----
        # 用 pop 而非 索引: 原地从 dict 移除, 保证后面 **batch 不会把 labels 误传给 forward
        labels = batch.pop("labels")
        extra_labels = {}
        if "audio_labels" in batch:
            extra_labels["audio_labels"] = batch.pop("audio_labels")

        # ---- 3) 模型前向 ----
        # **batch 解包: 各模型的 forward 签名不同, 由各自的 data_generator 保证字段对齐
        output = self.model(**batch)

        # ---- 4) 计算损失 ----
        loss_dict = self.loss_computer.compute(output, labels, **extra_labels)

        # ---- 5) 反向传播 ----
        loss_dict["total_loss"].backward()

        # ---- 6) 梯度裁剪: 防梯度爆炸, 尤其是 Transformer 训练前几百步 ----
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # ---- 7) 参数更新 → 学习率调度 → 清零梯度 ----
        # 顺序约定: optimizer.step 必须先于 scheduler.step (PyTorch >= 1.1)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self._global_step += 1

        # 汇总 metrics: 保留 tensor, 延迟同步到日志时刻
        metrics: Dict[str, Metric] = dict(loss_dict)
        metrics["grad_norm"] = grad_norm
        metrics["lr"] = self.scheduler.get_last_lr()[0]

        return metrics

    def train(self, num_steps: int = 0) -> List[Dict[str, float]]:
        """
        执行完整训练循环。

        Args:
            num_steps: 训练步数; 传 0 则使用 config.num_steps (默认值)。

        Returns:
            按 log_interval 采样的 metrics 列表 (已转为 float, 便于序列化/绘图)。
        """
        steps = num_steps if num_steps > 0 else self.config.num_steps
        all_metrics: List[Dict[str, float]] = []

        # 训练横幅
        print(f"\n{'=' * 60}")
        print(f"开始训练 | 总步数: {steps} | LR: {self.config.learning_rate}")
        print(f"{'=' * 60}")

        for step in range(1, steps + 1):
            metrics = self.train_step()

            # 仅在 第一步 / 间隔点 / 最后一步 打印, 减少日志噪声与 GPU 同步开销
            if step == 1 or step % self.config.log_interval == 0 or step == steps:
                scalar_metrics = self._to_scalar(metrics)
                self._log_metrics(step, steps, scalar_metrics)
                all_metrics.append(scalar_metrics)

        # 训练结束横幅, 顺带打印 loss 收敛情况, 方便快速判断训练是否生效
        print(f"{'=' * 60}")
        print("训练完成!")
        if len(all_metrics) >= 2:
            first_loss = all_metrics[0]["total_loss"]
            final_loss = all_metrics[-1]["total_loss"]
            print(f"Loss: {first_loss:.4f} -> {final_loss:.4f}")
        print(f"{'=' * 60}\n")

        return all_metrics

    @staticmethod
    def _to_scalar(metrics: Dict[str, Metric]) -> Dict[str, float]:
        """
        把 tensor 形式的 metrics 转为 float, 仅在打印日志时调用。

        统一 .item() 在此处发生, 让 train_step 内部全程在 GPU 上跑,
        避免每步同步带来的吞吐损失。
        """
        out: Dict[str, float] = {}
        for k, v in metrics.items():
            out[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        return out

    def _log_metrics(self, step: int, total_steps: int, metrics: Dict[str, float]):
        """
        格式化打印训练指标。

        - lr 用科学计数法 (跨多个量级时更易读);
        - 其余 metric 保留 4 位小数。
        """
        parts = [f"Step [{step:>4d}/{total_steps}]"]
        for k, v in metrics.items():
            if k == "lr":
                parts.append(f"{k}: {v:.2e}")
            else:
                parts.append(f"{k}: {v:.4f}")
        print(" | ".join(parts))
