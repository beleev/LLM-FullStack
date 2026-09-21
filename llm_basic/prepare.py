"""
prepare.py — 一次性脚本：下载 Tiny Shakespeare → 字符级编码 → train.bin / val.bin / meta.npz。

为什么单独一步：把"下载 + 建词表 + 编码"和训练循环解耦。
  - train.py 用 np.memmap 直接映射 .bin（每个 token 1 字节 uint8），不解析文本，启动瞬间完成
  - 词表只构造一次存进 meta.npz，train.py / sample.py / tokenizer.py 共用，保证 id 一致
关键数字：1,115,394 字符，vocab=65，90/10 切分 → train 1,003,854 / val 111,540 tokens。
读代码时盯住：stoi —— 整个"tokenizer 训练"就是 sorted(set(text)) 这一行。

用法：cd llm_basic && python prepare.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

HERE = Path(__file__).parent
INPUT_TXT = HERE / "input.txt"
TRAIN_BIN = HERE / "train.bin"
VAL_BIN = HERE / "val.bin"
META_NPZ = HERE / "meta.npz"


def download_if_missing() -> str:
    """下载原始文本（约 1MB），已存在则跳过。返回文本字符串。"""
    if not INPUT_TXT.exists():
        print(f"downloading Tiny Shakespeare from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, INPUT_TXT)
    text = INPUT_TXT.read_text(encoding="utf-8")
    print(f"loaded {len(text):,} characters from {INPUT_TXT.name}")
    return text


def main() -> None:
    text = download_if_missing()

    # 字符级词表：所有出现过的字符按字典序排序
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"vocab_size = {vocab_size}")
    print(f"chars: {''.join(chars)!r}")

    # 字符 → id。vocab=65 < 256，uint8 够用（文件大小 = token 数）
    assert vocab_size < 256, "vocab too large for uint8 — switch to uint16"
    data = np.array([stoi[c] for c in text], dtype=np.uint8)

    # 90/10 切分：按顺序切而不是随机抽，val 才是模型完全没见过的连续文本
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)
    print(f"train: {len(train):,} tokens → {TRAIN_BIN.name}")
    print(f"val:   {len(val):,} tokens → {VAL_BIN.name}")

    np.savez(
        META_NPZ,
        vocab_size=np.int32(vocab_size),
        chars=np.array(chars),  # numpy 会保存为定长 unicode 字符串数组
    )
    print(f"meta saved to {META_NPZ.name}")


if __name__ == "__main__":
    main()
