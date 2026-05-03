"""
偏好数据生成器 (DPO 专用)
=============================

为教学合成 (prompt, chosen, rejected) 三元组的最小数据生成器。

为什么共享 prompt?
    DPO 的 loss 是对同一 prompt 下两条不同 response 求 log-prob 差。
    构造数据时让两条样本共享 prompt 区, 避免引入额外 confounding (例如
    "如果 prompt 也不同, 模型学到的可能是 prompt 偏好而非 response 偏好")。
    这与真实数据集 (Anthropic HH-RLHF, UltraFeedback) 的格式一致。

约定:
    chosen_input_ids   = [prompt | chosen_response]
    rejected_input_ids = [prompt | rejected_response]
    chosen_labels      = chosen_input_ids 右移一位, prompt 段置 -100
    rejected_labels    = 同上, response 不同
"""

from typing import Dict, Optional

import torch

from llm_models.training.data import SyntheticDataGenerator


class PreferenceDataGenerator(SyntheticDataGenerator):
    """
    DPO 用偏好对合成数据生成器。

    每步产出一个 batch (4 个张量):
        chosen_input_ids:    [B, T]  prompt + 偏好回复
        rejected_input_ids:  [B, T]  prompt + 拒绝回复
        chosen_labels:       [B, T]  prompt 段 = -100, response 段 = 真实 next-token
        rejected_labels:     [B, T]

    Args:
        vocab_size:   词表大小, >= 2。
        batch_size:   批次大小。
        seq_len:      总长度 (prompt + response)。
        prompt_len:   prompt 前缀长度; None → seq_len // 2。
        device:       张量设备。
        seed:         独立随机源种子。
        fixed_batch:  True 时仅在初始化采样一次, 之后每步返回**同一份**偏好对。
                      教学场景默认开启: 让模型反复看到 (chosen, rejected) 同一对样本,
                      DPO 才能稳定地把 chosen 的 log-prob 拉高、rejected 拉低。
    """

    def __init__(
        self,
        vocab_size: int,
        batch_size: int = 2,
        seq_len: int = 32,
        prompt_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        fixed_batch: bool = True,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size 至少为 2")
        if seq_len < 2:
            raise ValueError("seq_len 至少为 2")
        if prompt_len is None:
            prompt_len = seq_len // 2
        if not (0 < prompt_len < seq_len):
            raise ValueError(
                f"prompt_len 必须在 (0, seq_len) 内, 当前 {prompt_len} / {seq_len}"
            )

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.device = device if device is not None else torch.device("cpu")
        self._gen: Optional[torch.Generator] = None
        if seed is not None:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(seed)

        self.fixed_batch = fixed_batch
        # 教学固定模式: 预先采样一份, 让 DPO 反复在同一对上优化以观察 reward gap 拉开
        self._cached_batch: Optional[Dict[str, torch.Tensor]] = (
            self._sample() if fixed_batch else None
        )

    def _rand_tokens(self, *shape: int) -> torch.Tensor:
        """从 [1, vocab_size) 采样 long 张量, 与 InstructionDataGenerator 同样
        把 0 留给 pad。"""
        return torch.randint(
            low=1,
            high=self.vocab_size,
            size=tuple(shape),
            device=self.device,
            generator=self._gen,
        )

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        # Trainer/DPOTrainer 会 pop 字段, 直接返回缓存会让第二步丢失 keys → 浅拷贝
        if self._cached_batch is not None:
            return dict(self._cached_batch)
        return self._sample()

    def _sample(self) -> Dict[str, torch.Tensor]:
        B, T, P = self.batch_size, self.seq_len, self.prompt_len
        R = T - P  # response 段长度

        # ---- 构造 prompt 段: chosen 与 rejected 共享前 P 个 token ----
        prompt = self._rand_tokens(B, P)

        # ---- 构造两条 response: 长度 R, 内容彼此独立 ----
        chosen_resp = self._rand_tokens(B, R)
        rejected_resp = self._rand_tokens(B, R)

        # 拼接成完整的输入序列, 长度 = T
        chosen_full = torch.cat([prompt, chosen_resp], dim=1)        # [B, T]
        rejected_full = torch.cat([prompt, rejected_resp], dim=1)    # [B, T]

        # ---- 构造 labels: 把 input 右移一位 ----
        # 第 t 位的标签 = input 的第 t+1 位; 最后一位 t=T-1 没有 next-token, 用占位
        # 占位会落在 response 区, 但因为 t=T-1 处 labels 实际有效, 我们用一个随机
        # 合法 token; 它会被 loss 正常统计 (response 区不 mask)。
        tail = self._rand_tokens(B, 1)  # [B, 1]  最末位置的 next-token 占位
        chosen_seq = torch.cat([chosen_full, tail], dim=1)        # [B, T+1]
        rejected_seq = torch.cat([rejected_full, tail], dim=1)    # [B, T+1]

        chosen_labels = chosen_seq[:, 1:].clone()       # [B, T]
        rejected_labels = rejected_seq[:, 1:].clone()   # [B, T]

        # DPO 关键: prompt 段不算 log-prob, 只统计 response 部分
        # 注: 第 P-1 位的目标 (即 response 的第一位) 也归入 prompt mask, 因为我们
        # 关心的是 response 内部的生成概率, 不关心 "prompt 终点 → response 起点"
        # 这一步的转移 (它对所有候选回复完全一致, 无法区分 chosen vs rejected)
        chosen_labels[:, :P] = -100
        rejected_labels[:, :P] = -100

        return {
            "chosen_input_ids": chosen_full,
            "rejected_input_ids": rejected_full,
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
        }
