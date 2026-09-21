"""
Prompt 数据 (GRPO / on-policy 蒸馏 / RFT 等**在线**方法)

与 SFT / DPO 的根本区别: 数据集里只有 prompt。回复由当前策略现场采样, 分数由 verifier 现场给 ——
训练数据的分布随策略一起漂移 (on-policy), 这也是在线方法没法塞进 "取 batch → 算 loss" 这个 Trainer 的原因。
"""

from typing import Dict

import torch

from llm_models.training.data import SyntheticDataGenerator
from llm_finetune.data.tasks import SeqTask


class PromptDataGenerator(SyntheticDataGenerator):
    def __init__(self, task: SeqTask, batch_size: int = 16, split: str = "train",
                 fixed: bool = False) -> None:
        self.task, self.batch_size, self.split, self.fixed = task, batch_size, split, fixed

    def _sample(self) -> Dict[str, torch.Tensor]:
        return {"prompts": self.task.sample_prompts(self.batch_size, self.split)}   # [B, P]
