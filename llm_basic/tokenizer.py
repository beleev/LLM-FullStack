"""
tokenizer.py — 字符级 tokenizer：一个字符 = 一个 token，词表来自 prepare.py 写的 meta.npz。

训练流水线（prepare / train / sample）用的就是它。最朴素的选择：词表只有 65，
encode/decode 就是查表，没有任何要"训练"的东西。代价是序列长（一个单词 5~10 个 token，
而注意力是 O(T²)），且模型得自己学拼写 —— bpe.py 演示了怎么解决，但本目录的训练并未接入它。
读代码时盯住：stoi / itos 这两张互逆的表。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CharTokenizer:
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
