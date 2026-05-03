"""
tokenizer.py — 字符级 tokenizer (零依赖)

为什么用字符级?
    - 教学场景, 词表只需 ~100 个字符, 模型小到 CPU 秒级跑完
    - 不依赖 sentencepiece / tiktoken, 一个文件搞定
    - 所有现代 tokenizer (BPE, WordPiece) 的接口都是 encode/decode,
      用字符级演示语义完全等价

特殊 token:
    - <pad> = 0   填充, 推理一般用不到, 留位置给 batched padding
    - <bos> = 1   sequence 起始
    - <eos> = 2   sequence 终止 (生成停止信号)
"""
from __future__ import annotations
from typing import List


class CharTokenizer:
    """字符级 tokenizer。

    词表 = 所有可打印 ASCII + 中文常用字 (按需扩) + 3 个特殊 token。
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self, extra_chars: str = ""):
        # 基本词表: 可见 ASCII (32~126) + 换行 + tab
        ascii_chars = "".join(chr(c) for c in range(32, 127)) + "\n\t"
        chars = list(dict.fromkeys(ascii_chars + extra_chars))  # 去重保序
        self.itos: List[str] = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    # --- 编码 / 解码 ------------------------------------------------- #

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """字符串 → token id 列表。未识字符跳过 (教学简化)。"""
        ids = [self.stoi[c] for c in text if c in self.stoi]
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """token id 列表 → 字符串。"""
        out = []
        for i in ids:
            tok = self.itos[i]
            if skip_special and i in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            out.append(tok)
        return "".join(out)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab={self.vocab_size})"


if __name__ == "__main__":
    tok = CharTokenizer()
    text = "Hello, vLLM!"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    print(f"text  = {text!r}")
    print(f"ids   = {ids}")
    print(f"back  = {tok.decode(ids)!r}")
    print(f"vocab = {len(tok)}")
