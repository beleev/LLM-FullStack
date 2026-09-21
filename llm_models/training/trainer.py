"""
通用训练器 — 一个训练循环跑全库所有模型

是什么: Trainer 只认两个策略接口 —— SyntheticDataGenerator (batch 怎么来) 与 LossComputer (loss 怎么算);
        新模型只需配一对 generator / loss, 循环本身不改。
单步: generate_batch → pop labels → model(**batch) → loss → backward → clip_grad_norm → AdamW.step → scheduler.step
关键数字: 第 1 步的 lr = 0 (线性 warmup 从 0 起) → 日志里第一条 loss 就是 "未训练模型" 的 loss,
          train 脚本据此断言 `|loss₁ − ln V| < 0.5`。
注意: 默认数据是固定的一个 batch (见 data.py), 所以 "loss 下降" = 能背下这个 batch。
读代码时盯住: `batch.pop("labels")` —— dict 里剩下的 key 必须正好是 model.forward 的形参名。
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
    AdamW + (linear warmup → cosine) + 梯度裁剪。

    - AdamW: weight decay 与梯度解耦, 不被 Adam 的自适应缩放扭曲 (严格做法应排除 norm/bias, 教学版统一处理)。
    - clip_grad_norm_: Transformer 训练初期常有梯度尖峰 (softmax 饱和), 一次大更新就能毁掉模型。
    - metrics 以 tensor 返回, 只在打日志时 .item(): 每步 .item() 会强制 GPU→CPU 同步。
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
