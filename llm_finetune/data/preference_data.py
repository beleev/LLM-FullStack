"""
偏好数据 (DPO / SimPO / ORPO / Reward Model 共用)

是什么: (prompt, chosen = 正确回复, rejected = 损坏回复)。两条序列共享 prompt, 只有回复不同,
        所以 log π(y_w|x) − log π(y_l|x) 衡量的纯粹是 "对回复的偏好"。
rejected 有一半比 chosen 短一个 token → 右 pad → 同时给出 attention_mask (reward model 要用它找最后一个真 token)。
Trainer 约定 batch 里必须有 "labels": 这里它是 dict {"chosen": …, "rejected": …}, 原样交给 loss。
"""

from typing import Dict

import torch

from llm_models.training.data import SyntheticDataGenerator
from llm_finetune.data.tasks import PAD, SeqTask, make_labels


class PreferenceDataGenerator(SyntheticDataGenerator):
    def __init__(self, task: SeqTask, batch_size: int = 64, split: str = "train",
                 fixed: bool = False) -> None:
        self.task, self.batch_size, self.split, self.fixed = task, batch_size, split, fixed

    def _sample(self) -> Dict[str, torch.Tensor]:
        P = self.task.prompt_len
        prompts = self.task.sample_prompts(self.batch_size, self.split)       # [B, P]
        good = self.task.target(prompts)                                      # [B, R]
        bad = self.task.corrupt(good)                                         # [B, R] 可能含右 pad
        batch, labels = {}, {}
        for name, resp in (("chosen", good), ("rejected", bad)):
            seq = torch.cat([prompts, resp], dim=1)                           # [B, P+R]
            idx, labels[name] = make_labels(seq, P)                           # [B, P+R]
            batch[f"{name}_input_ids"] = idx
            batch[f"{name}_attention_mask"] = (idx != PAD).long()             # 1 = 真 token
        batch["labels"] = labels
        return batch
