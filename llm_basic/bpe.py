"""
bpe.py — 手写一个最小 byte-level BPE tokenizer（GPT-2 同款思想）

字符级 tokenizer（tokenizer.py）的词表 = 语料里出现过的字符，简单但有两个代价：
  1. 序列太长：一个英文单词要 5~10 个 token，注意力是 O(T^2)，长序列很贵
  2. 没有"词"的概念：模型要花参数容量自己学字母如何拼成词

BPE (Byte Pair Encoding) 用一句话概括：
    从 256 个字节出发，反复把「出现频率最高的相邻 token 对」合并成一个新 token，
    直到词表到达目标大小。

  - 训练阶段：统计 → 合并 → 再统计 → 再合并（贪心，无任何梯度）
  - 编码阶段：对新文本按「合并的先后顺序」重放这些规则
  - 解码阶段：查表把 token 还原成字节再拼回字符串（永远无损）

这就是 GPT-2 / GPT-3 / Qwen 词表的构造方式，LLaMA 的 SentencePiece-BPE 思想相同。
真实实现还有 regex 预切分、special tokens、并行化等工程细节，这里全部剥掉。

运行:
    python bpe.py                  # 在 input.txt 前 64KB 上训练 300 次合并
    python bpe.py --merges 500     # 自定义合并次数
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


# --------------------------------------------------------------------- #
# 训练：统计相邻对 → 合并最高频对，循环 num_merges 次                    #
# --------------------------------------------------------------------- #

def get_pair_counts(ids: list[int]) -> Counter:
    """统计序列中每个相邻 (a, b) token 对出现的次数。"""
    return Counter(zip(ids, ids[1:]))


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把序列中所有相邻的 pair=(a,b) 替换成 new_id，返回新序列（不改原序列）。"""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2  # 跳过被合并的两个
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> dict[tuple[int, int], int]:
    """
    在 text 上训练 BPE，返回 merges 表：{(a, b) -> 合并后的新 token id}。

    新 id 从 256 开始递增 —— dict 的插入顺序就是合并的优先级，
    编码时必须按同样的顺序重放（Python 3.7+ dict 保序）。
    """
    ids = list(text.encode("utf-8"))  # 起点：每个字节一个 token，id ∈ [0, 256)
    merges: dict[tuple[int, int], int] = {}
    for step in range(num_merges):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair = counts.most_common(1)[0][0]   # 最高频的相邻对
        new_id = 256 + step
        ids = merge_pair(ids, pair, new_id)
        merges[pair] = new_id
    return merges


def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """从 merges 表展开每个 token id 对应的原始字节串（解码查表用）。"""
    vocab = {i: bytes([i]) for i in range(256)}
    # merges 按插入顺序遍历：合并 (a, b) 时 a、b 一定已经在 vocab 里
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


# --------------------------------------------------------------------- #
# 编码 / 解码                                                            #
# --------------------------------------------------------------------- #

def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """
    对新文本重放训练时的合并规则。

    每轮找出当前序列中「训练时最早被合并」的那个相邻对（rank 最小），
    优先合并它 —— 顺序必须与训练一致，否则同一个词会切出不同 token。
    """
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        counts = get_pair_counts(ids)
        # 在当前出现的相邻对里，挑训练时 rank 最小（最早合并）的
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # 所有相邻对都不在 merges 表里，编码结束
        ids = merge_pair(ids, pair, merges[pair])
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """token id 序列 → 字节串 → 字符串。errors='replace' 兜底非法 UTF-8 边界。"""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


# --------------------------------------------------------------------- #
# 演示                                                                   #
# --------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="最小 byte-level BPE 演示")
    parser.add_argument("--merges", type=int, default=300, help="合并次数 (词表 = 256 + merges)")
    parser.add_argument("--train-bytes", type=int, default=64 * 1024, help="用于训练的语料字节数")
    args = parser.parse_args()

    data_path = Path(__file__).parent / "input.txt"
    if not data_path.exists():
        raise SystemExit("找不到 input.txt，请先运行: python prepare.py")
    corpus = data_path.read_text(encoding="utf-8")[: args.train_bytes]

    print(f"语料: {len(corpus):,} 字符 | 目标: {args.merges} 次合并 (词表 256 → {256 + args.merges})")
    merges = train_bpe(corpus, num_merges=args.merges)
    vocab = build_vocab(merges)

    # ---- 1) 看看模型学到了什么样的"词" ------------------------------ #
    print("\n[1] 前 12 次合并（最高频的相邻对先被合并）")
    for i, ((a, b), new_id) in enumerate(merges.items()):
        if i >= 12:
            break
        print(f"  merge {i:>3}: {vocab[a]!r} + {vocab[b]!r} -> {vocab[new_id]!r}  (id={new_id})")

    longest = sorted(vocab.values(), key=len, reverse=True)[:8]
    print("\n[2] 词表里最长的 8 个 token（频繁词组被自动发现）")
    print("  " + "  ".join(repr(t.decode('utf-8', errors='replace')) for t in longest))

    # ---- 2) 编码 / 解码闭环 + 压缩率 -------------------------------- #
    sample = "First Citizen:\nBefore we proceed any further, hear me speak."
    ids = encode(sample, merges)
    roundtrip = decode(ids, vocab)
    assert roundtrip == sample, "BPE 解码必须无损还原"

    print("\n[3] 编码示例")
    print(f"  原文 ({len(sample)} 字符): {sample!r}")
    shown = [vocab[i].decode("utf-8", errors="replace") for i in ids[:18]]
    print(f"  前 18 个 token: {shown}")
    print(f"  字符级需要 {len(sample)} 个 token, BPE 只要 {len(ids)} 个")
    print(f"  压缩率: {len(sample) / len(ids):.2f} 字符/token  (GPT-2 在英文上约 4)")

    print("\n  OK: 词表大小是「序列长度 vs embedding 参数量」之间的工程权衡，")
    print("      BPE 用一个贪心统计算法把这个权衡变成了可调的旋钮 (--merges)。")


if __name__ == "__main__":
    main()
