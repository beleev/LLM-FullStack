"""
指令数据生成器 (SFT / LoRA 共用)
==================================

为教学合成 (prompt, response) 配对的最小数据生成器。真实 SFT 需要从
Alpaca / ShareGPT 等数据集读取 jsonl 后用 tokenizer 编码; 这里用随机 token
模拟即可, 重点在于 **labels 中 prompt 段被置 -100**, 这是 SFT 与
pre-training 在数据形态上唯一的、本质的差异。

teacher-forcing 与 shift-by-one 标签 (与预训练相同):
    模型在位置 t 看到的输入是 token_t, 它要预测 token_{t+1}。
    实现上, 我们生成长度 seq_len + 1 的随机序列 X, 切片成:
        input  = X[:, :-1]    (长度 seq_len)
        labels = X[:,  1:]    (每位置 t 的目标 = 下一位 token)

SFT 独有的步骤:  prompt mask
    取一个 prompt_len < seq_len, 把 labels 的前 prompt_len 个位置改写成 -100。
    cross_entropy(ignore_index=-100) 会跳过它们, 等价于 "loss 只在 response 上算"。
    这是把 LM 训成"指令跟随者"而非"通用续写器"的关键。
"""

from typing import Dict, Optional

import torch

from llm_models.training.data import SyntheticDataGenerator


class InstructionDataGenerator(SyntheticDataGenerator):
    """
    SFT / LoRA 微调用合成数据生成器。

    每步产出一个 batch:
        idx:    [B, T]  输入 token id
        labels: [B, T]  目标 token id, 其中 prompt 区 (前 prompt_len 位) 为 -100

    Args:
        vocab_size:   词表大小, 必须 >= 2 (1 留给 BOS / 占位)。
        batch_size:   批次大小。
        seq_len:      输入长度 (= input.size(1) = labels.size(1))。
        prompt_len:   prompt 占总长的前缀长度。response_len = seq_len - prompt_len。
                      None 时默认取 seq_len // 2, 一半 prompt 一半 response。
        device:       张量所在设备。
        seed:         固定后端的 generator, 便于复现。None 表示沿用全局随机源。
        fixed_batch:  True 时仅在初始化采样一次, 之后每步返回**同一份**数据。
                      教学场景默认开启: 在合成的"假指令"上重复训练相当于让模型
                      记忆这条样本, 能直观地看到 loss 单调下降, 证明 finetune
                      通路可工作。生产中应关闭, 由真实 DataLoader 替代。
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
            raise ValueError("vocab_size 至少为 2 (token id 从 1 开始)")
        if seq_len < 2:
            raise ValueError("seq_len 至少为 2, 才能切出 input/labels")
        # prompt_len 默认值: 一半 prompt 一半 response, 教学最直观
        if prompt_len is None:
            prompt_len = seq_len // 2
        if not (0 < prompt_len < seq_len):
            raise ValueError(
                f"prompt_len 必须在 (0, seq_len) 区间内, 当前 {prompt_len} / {seq_len}"
            )

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.prompt_len = prompt_len
        self.device = device if device is not None else torch.device("cpu")

        # 用独立 generator 而非全局随机, 保证多生成器并存时互不干扰
        self._gen: Optional[torch.Generator] = None
        if seed is not None:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(seed)

        self.fixed_batch = fixed_batch
        # 教学模式下预先采样一份并缓存, 让每步训练看到同一条样本 → 可被记忆
        self._cached_batch: Optional[Dict[str, torch.Tensor]] = (
            self._sample() if fixed_batch else None
        )

    def _sample(self) -> Dict[str, torch.Tensor]:
        """新采样一个 batch (内部使用)。"""
        # token id 从 1 开始: 0 通常作为 pad, 避免合成数据撞上 pad token
        # 长度 +1: 用最后一位作为最末位置的 next-token 目标
        x = torch.randint(
            low=1,
            high=self.vocab_size,
            size=(self.batch_size, self.seq_len + 1),
            device=self.device,
            generator=self._gen,
        )

        # teacher forcing: 输入和标签错位一格
        idx = x[:, :-1].contiguous()       # [B, T]
        labels = x[:, 1:].clone()          # [B, T]   先复制再写 mask, 不污染 idx

        # SFT 关键: prompt 段标签置 -100, 让 cross_entropy 跳过这些位置
        labels[:, : self.prompt_len] = -100

        return {"idx": idx, "labels": labels}

    def generate_batch(self) -> Dict[str, torch.Tensor]:
        """
        生成一个 batch。返回字段:
            idx:    模型 forward 的输入
            labels: 给 SFTLoss / StandardLMLoss 用的目标 (prompt 段 = -100)
        """
        # 教学固定模式: 复用缓存张量, 但返回 dict 的浅拷贝
        # ——Trainer.train_step 会 pop("labels"), 直接返回原 dict 会破坏缓存。
        if self._cached_batch is not None:
            return dict(self._cached_batch)
        return self._sample()
