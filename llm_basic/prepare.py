"""
prepare.py — 一次性脚本：下载 Tiny Shakespeare，做字符级编码，存为 train.bin / val.bin / meta.npz

为什么单独一个 prepare 步骤？
  把"下载 + 词表构建 + 编码"和"训练循环"解耦：
    - 训练时只需 mmap 二进制 token 序列，启动飞快
    - 词表 (chars) 只构造一次，sample.py 也直接复用
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

    # 编码整篇文本：字符 → id
    # Tiny Shakespeare vocab_size = 65 < 256，uint8 够用
    assert vocab_size < 256, "vocab too large for uint8 — switch to uint16"
    data = np.array([stoi[c] for c in text], dtype=np.uint8)

    # 90/10 切分
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)
    print(f"train: {len(train):,} tokens → {TRAIN_BIN.name}")
    print(f"val:   {len(val):,} tokens → {VAL_BIN.name}")

    # 词表也存下来，sample.py / train.py 都要读
    np.savez(
        META_NPZ,
        vocab_size=np.int32(vocab_size),
        chars=np.array(chars),  # numpy 会保存为定长 unicode 字符串数组
    )
    print(f"meta saved to {META_NPZ.name}")


if __name__ == "__main__":
    main()
