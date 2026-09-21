"""
指令数据 (SFT / LoRA / DoRA / QLoRA 共用)

是什么: 每步从 SeqTask 采一批 (prompt, 正确回复), 返回 {"idx", "labels"}; prompt 段 label = -100。
与预训练数据的唯一差别就是这个 mask (见 tasks.make_labels); 缓存 / fixed 逻辑继承自
llm_models.training.data.SyntheticDataGenerator, 这里只实现 `_sample()`。
默认 fixed=False: 任务有规律可学, 每步换新 batch, 成功标准是留出集 exact-match 而不是 "背下一个 batch"。
"""

from typing import Dict

import torch

from llm_models.training.data import SyntheticDataGenerator
from llm_finetune.data.tasks import SeqTask, make_labels


class InstructionDataGenerator(SyntheticDataGenerator):
    def __init__(self, task: SeqTask, batch_size: int = 64, fixed: bool = False) -> None:
        self.task, self.batch_size, self.fixed = task, batch_size, fixed

    def _sample(self) -> Dict[str, torch.Tensor]:
        prompts = self.task.sample_prompts(self.batch_size, "train")          # [B, P]
        seq = torch.cat([prompts, self.task.target(prompts)], dim=1)          # [B, P+R]
        idx, labels = make_labels(seq, self.task.prompt_len)                  # [B, P+R] ×2
        return {"idx": idx, "labels": labels}
