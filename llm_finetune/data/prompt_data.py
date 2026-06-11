"""
Prompt 数据生成器 (GRPO / 在线 RL 专用)
=========================================

与 SFT/DPO 的数据形态根本不同: 在线 RL 阶段**没有标注好的 response**,
数据集里只有 prompt —— response 由当前 policy 实时采样, 奖励由
reward_fn (规则验证器或 reward model) 实时计算。

这也是 GRPO/PPO 与 DPO 的本质区别之一:
    DPO:  离线偏好对 (prompt, chosen, rejected), 数据固定
    GRPO: 在线采样, 数据分布随 policy 变化而漂移 (on-policy)
"""

from typing import Dict, Optional

import torch

from llm_models.training.data import SyntheticDataGenerator


class PromptDataGenerator(SyntheticDataGenerator):
    """
    每步产出一批随机 prompt:
        prompts: [B, P]  token id ∈ [1, vocab_size)

    Args:
        vocab_size:  词表大小
        batch_size:  每步 prompt 条数 (注意: 实际采样条数 = B × group_size)
        prompt_len:  prompt 长度 P
        seed:        独立随机源
        fixed_batch: True 时每步返回同一批 prompt (教学默认, 收敛可观察)
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int = 4,
        prompt_len: int = 4,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        fixed_batch: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.device = device if device is not None else torch.device("cpu")

        self._gen: Optional[torch.Generator] = None
        if seed is not None:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(seed)

        self.fixed_batch = fixed_batch
        self._cached: Optional[Dict[str, torch.Tensor]] = (
            self._sample() if fixed_batch else None
        )

    def _sample(self) -> Dict[str, torch.Tensor]:
        prompts = torch.randint(
            low=1,
            high=self.vocab_size,
            size=(self.batch_size, self.prompt_len),
            device=self.device,
            generator=self._gen,
        )
        return {"prompts": prompts}

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        if self._cached is not None:
            return dict(self._cached)
        return self._sample()
