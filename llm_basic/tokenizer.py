"""
tokenizer.py — 字符级编码器，从 prepare.py 生成的 meta.npz 读取词表。

字符级 tokenizer 是最朴素的选择：
  - 词表小（Tiny Shakespeare 65 个字符）
  - 不需要 BPE 训练
  - 编码 / 解码就是查表
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CharTokenizer:
    """不可变 dataclass：构造完就只读。"""
    chars: tuple[str, ...]
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_meta(cls, meta_path: str | Path) -> "CharTokenizer":
        """从 prepare.py 写入的 meta.npz 加载。"""
        meta = np.load(meta_path, allow_pickle=False)
        chars = tuple(str(c) for c in meta["chars"])
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        return cls(chars=chars, stoi=stoi, itos=itos)

    def encode(self, text: str) -> np.ndarray:
        """字符串 → uint8 ids 数组。遇到未知字符直接报错。"""
        return np.array([self.stoi[c] for c in text], dtype=np.uint8)

    def decode(self, ids: np.ndarray | list[int]) -> str:
        """ids 序列 → 字符串。"""
        return "".join(self.itos[int(i)] for i in ids)
