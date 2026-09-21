"""
m14 demo — Structured Output: FSM 约束解码, 从字符级到多字符 token 级 mask 表

[1] 字符级: 随机 logits + JsonFSM, 200 个 seed 全部 json.loads 通过且 pair 数 ∈ [1, max_pairs]
[2] token 级: 多字符词表 (≤128) 预编译 mask_table / next_state, 驱动真实 TinyLM 的 logits
[3] 无约束对照: 同一个模型几乎吐不出合法 JSON
[4] max_tokens: 只 mask 会被截断; 用 need 表做预算感知的强制收尾
[5] 查表 O(1) vs 在线现算 O(V·len) 的每步耗时

实现见 grammar.py。运行: python -m llm_infer.m14_structured_output.demo
"""
from __future__ import annotations

import json
import time

import numpy as np

from llm_infer.core import CharTokenizer, ModelConfig, TinyLM
from llm_infer.core.utils import banner, kv
from llm_infer.m14_structured_output.grammar import (
    JsonFSM, State, build_vocab, compile_char_dfa, compile_token_table, generate, token_row)

MAX_PAIRS = 3


def parse_pairs(s: str):
    """合法 JSON object → pair 列表 (用 list 钩子, 重复 key 不会被 dict 吞掉); 否则 None。"""
    try:
        obj = json.loads(s, object_pairs_hook=list)
    except ValueError:
        return None
    return obj if isinstance(obj, list) and s.lstrip().startswith("{") else None


def char_level_decode(tok: CharTokenizer, seed: int) -> str:
    """字符级词表 + 随机 logits: 每步 legal_chars() → mask → argmax → advance。"""
    rng, fsm, out = np.random.RandomState(seed), JsonFSM(max_pairs=MAX_PAIRS), []
    while fsm.state != State.DONE:
        logits = rng.randn(tok.vocab_size)                       # (V,) 任意 logits
        legal = [tok.stoi[c] for c in fsm.legal_chars()]
        masked = np.full(tok.vocab_size, -np.inf)
        masked[legal] = logits[legal]
        ch = tok.itos[int(np.argmax(masked))]
        fsm.advance(ch)
        out.append(ch)
    return "".join(out)


def main() -> None:
    banner("M14 - Structured Output (FSM → token mask table)")

    # ---- [1] 字符级 FSM ------------------------------------------------
    print("\n[1] 字符级 FSM + 随机 logits (200 个 seed)")
    tok = CharTokenizer()
    outs = [char_level_decode(tok, seed) for seed in range(200)]
    for s in outs[:3]:
        print(f"    {s}")
    n_pairs = [len(parse_pairs(s) or []) for s in outs]
    hist = {n: n_pairs.count(n) for n in sorted(set(n_pairs))}
    kv("pair 数分布 {pairs: 次数}", hist)
    assert all(1 <= n <= MAX_PAIRS for n in n_pairs), "每个输出都应是 1..max_pairs 对的合法 JSON"
    assert set(hist) == {1, 2, 3}, "1 对也应出现 (两个分支若用不同的计数, string 结尾会被迫 ≥2 对)"

    # ---- [2] token 级预编译 --------------------------------------------
    print("\n[2] 多字符词表 → 预编译 mask_table[state] / next_state[state, token]")
    vocab = build_vocab()
    t0 = time.perf_counter()
    table = compile_token_table(JsonFSM(max_pairs=MAX_PAIRS), vocab)
    compile_ms = (time.perf_counter() - t0) * 1e3
    S, V = table.next_state.shape
    kv("词表 V (多字符 token 数)", f"{V} ({sum(len(p) > 1 for p in vocab[1:])})")
    kv("DFA 状态数 S", S)
    kv("表大小 S×V / 合法占比", f"{S * V} / {table.mask_table.mean():.1%}")
    kv("预编译耗时 (一次性)", f"{compile_ms:.1f} ms")
    t_true = vocab.index("true")
    kv("token 'true' 合法的状态数", f"{int(table.mask_table[:, t_true].sum())} (value 起始处, 以及 key/string 内还剩 ≥4 字符额度时)")

    lm = TinyLM(ModelConfig(vocab_size=V))                       # 随机权重: logits 是"任意"的
    N, MAX_TOKENS = 100, 40
    rng = np.random.RandomState(0)
    outs = [generate(lm, table, MAX_TOKENS, rng, "budget") for _ in range(N)]
    for s, _ in outs[:3]:
        print(f"    {s}")
    greedy, _ = generate(lm, table, MAX_TOKENS, rng, "budget", temperature=0)
    print(f"    greedy: {greedy}")
    n_ok = sum(parse_pairs(s) is not None and done for s, done in outs)
    kv(f"约束采样 {N} 次, 合法 JSON", f"{n_ok}/{N}")
    assert n_ok == N and parse_pairs(greedy) is not None
    assert all(1 <= len(parse_pairs(s)) <= MAX_PAIRS for s, _ in outs)

    # ---- [3] 无约束对照 ------------------------------------------------
    print("\n[3] 同一个模型, 不加 mask")
    rng = np.random.RandomState(0)
    raw = [generate(lm, table, MAX_TOKENS, rng, "none")[0] for _ in range(N)]
    print(f"    {raw[0][:60]!r}")
    n_raw = sum(parse_pairs(s) is not None for s in raw)
    kv(f"无约束采样 {N} 次, 合法 JSON", f"{n_raw}/{N}")
    assert n_raw <= N * 0.05

    # ---- [4] max_tokens 与强制收尾 --------------------------------------
    print("\n[4] max_tokens 很紧时 (=12; 本 grammar 最短合法输出 = "
          f"{int(table.need[0].min()) + 1} 个 token 含 EOS)")
    res = {}
    for mode in ("mask", "budget"):
        rng = np.random.RandomState(1)
        runs = [generate(lm, table, 12, rng, mode) for _ in range(N)]
        res[mode] = sum(done and parse_pairs(s) is not None for s, done in runs)
        kv(f"mode={mode:<6} 完整合法 JSON", f"{res[mode]}/{N}   例: {runs[0][0]!r}")
    assert res["budget"] == N and res["mask"] < N

    # ---- [5] 每步开销: 查表 vs 现算 -------------------------------------
    print("\n[5] 每个 decode step 取 mask 的耗时 (遍历全部状态取平均)")
    trans, accept = compile_char_dfa(JsonFSM(max_pairs=MAX_PAIRS))
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        for s in range(S):
            on_the_fly = token_row(trans, accept, vocab, s) >= 0  # O(V·len): V 个 token 逐字符走
    fly_us = (time.perf_counter() - t0) / (reps * S) * 1e6
    t0 = time.perf_counter()
    for _ in range(reps):
        for s in range(S):
            looked_up = table.mask_table[s]                      # O(1): 取一行
    lut_us = (time.perf_counter() - t0) / (reps * S) * 1e6
    kv("在线现算 O(V·len)", f"{fly_us:.1f} µs/step")
    kv("查表 O(1)", f"{lut_us:.3f} µs/step  ({fly_us / lut_us:.0f}x)")
    kv("预编译摊销", f"≈ {compile_ms * 1e3 / fly_us:.0f} 个 step 的现算成本, 之后全是净赚")
    assert lut_us < fly_us
    print(f"\n  注: 现算成本 ∝ V。按 V 线性外推到 128k 词表 ≈ {fly_us * 128_000 / V / 1e3:.0f} ms/step (本 Python 实现;"
          "\n      C++ 快两个数量级也仍与一次 GPU decode 前向可比), 查表始终是取一行 bitmask。")


if __name__ == "__main__":
    main()
