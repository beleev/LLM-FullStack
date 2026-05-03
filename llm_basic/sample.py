"""
sample.py — 加载 ckpt.npz，自回归地生成字符。

用法:
    python sample.py                                # 用默认 prompt
    python sample.py "ROMEO:" --max-new 200         # 自定义 prompt + 长度
    python sample.py "" --temperature 0.8 --top-k 10

采样过程
========
给定上下文 ids:
    while 没生成够 max_new_tokens:
        # 只喂最后 max_seq_len 个 token（窗口右滑）
        logits = forward(W, ids[-T:])              # (1, T, V)
        next_logits = logits[0, -1] / temperature  # (V,)
        next_logits = top_k_filter(next_logits, k) # 屏蔽 top-k 之外
        probs = softmax(next_logits)
        next_id = multinomial(probs)
        ids.append(next_id)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import softmax, transformer_forward
from tokenizer import CharTokenizer

HERE = Path(__file__).parent
CKPT_NPZ = HERE / "ckpt.npz"
META_NPZ = HERE / "meta.npz"


def load_ckpt(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """与 train.py 的 save_ckpt 对应：拆出 _config_* 与参数。"""
    bundle = np.load(path, allow_pickle=False)
    config: dict[str, int] = {}
    W: dict[str, np.ndarray] = {}
    for k in bundle.files:
        if k.startswith("_config_"):
            config[k[len("_config_") :]] = int(bundle[k])
        else:
            W[k] = bundle[k]
    return W, config


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """把 logits 中除 top-k 之外的位置置 -inf（softmax 后概率 = 0）。"""
    if k <= 0 or k >= logits.size:
        return logits
    # 第 k 大的值
    threshold = np.partition(logits, -k)[-k]
    return np.where(logits < threshold, -np.inf, logits)


def generate(
    W: dict[str, np.ndarray],
    config: dict[str, int],
    prompt_ids: np.ndarray,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """自回归生成。返回完整 ids（含 prompt）。"""
    T_max = config["max_seq_len"]
    ids = [int(i) for i in prompt_ids]

    for _ in range(max_new_tokens):
        # 只看最后 T_max 个 token（KV 不缓存的最朴素做法 —— 每步重算整个 forward）
        ctx = np.array(ids[-T_max:], dtype=np.int64)[None, :]   # (1, T)
        logits, _ = transformer_forward(W, ctx)                 # (1, T, V)
        next_logits = logits[0, -1] / max(temperature, 1e-8)    # (V,)
        next_logits = top_k_filter(next_logits, top_k)
        probs = softmax(next_logits)                            # (V,)

        # 采样：rng.choice 按概率抽一个 id
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)

    return np.array(ids, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="?", default="\n",
        help="起始文本（默认换行符；空字符串时用换行起头）",
    )
    parser.add_argument("--max-new", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="0 = 关闭 top-k 过滤；>0 = 只保留概率最高的 k 个候选",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"loading {CKPT_NPZ.name} and {META_NPZ.name} ...")
    W, config = load_ckpt(CKPT_NPZ)
    tok = CharTokenizer.from_meta(META_NPZ)
    print(f"  config: {config}")

    prompt = args.prompt if args.prompt else "\n"
    prompt_ids = tok.encode(prompt)
    rng = np.random.default_rng(args.seed)

    out_ids = generate(
        W, config, prompt_ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )

    text = tok.decode(out_ids)
    print("─" * 60)
    print(text)
    print("─" * 60)


if __name__ == "__main__":
    main()
