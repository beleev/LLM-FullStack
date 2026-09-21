"""
full_engine demo — mini-vLLM 集成演示, 每一节都以 assert 收尾

运行: python -m llm_infer.full_engine.demo
    [1] 共享 system prompt 的 5 条请求: 看每步 batch 里 prefill chunk (P) 与 decode (D) 混跑, 前缀命中对账
    [2] 同一批 prompt 再来一遍: 上一轮的 block 已释放但仍可命中 (stale-entry 回归)
    [3] 极小 KV pool: 抢占 + 无活锁, 输出不丢不多
    [4] 同 batch 不同采样参数        [5] 随机配置 fuzz
契约: greedy 下每条输出与 TinyLM.generate_greedy 逐 token 相同。
"""
from __future__ import annotations
import numpy as np

from llm_infer.core import ModelConfig, CharTokenizer, TinyLM
from llm_infer.core.utils import banner, kv
from llm_infer.m10_sampling.samplers import SamplingParams
from llm_infer.full_engine.engine import Engine, EngineConfig

MAX_STEPS = 3000      # 超过即判活锁


def run_and_check(engine: Engine, prompts, max_new: int, verbose: bool = False):
    """跑完并断言每条输出 == 朴素 generate_greedy (遇 EOS 提前停)。"""
    sids = [engine.add_request(p, max_new=max_new) for p in prompts]
    last = None
    while engine.has_unfinished():
        done = engine.step()
        assert engine.steps < MAX_STEPS, "活锁"
        if verbose:
            desc = " ".join(f"s{s.seq_id}:{'P' + str(n) if n > 1 else 'D'}" for s, n in engine.last_batch)
            tail = "".join(f"  ✓s{s.seq_id} {engine.tok.decode(s.output_ids)!r}" for s in done)
            if desc == last and not done:
                continue                                  # batch 组成没变的 decode 步不重复打印
            last = desc
            print(f"  step {engine.steps:>3} [{desc:<26}] tokens={sum(n for _, n in engine.last_batch):>3}{tail}")
    for sid in sids:
        seq = engine.finished[sid]
        ref = [int(t) for t in engine.lm.generate_greedy(np.array(seq.prompt_ids), max_new)[len(seq.prompt_ids):]]
        out = seq.output_ids
        assert out == ref[:len(out)] and (len(out) == max_new or out[-1] == seq.eos_id), (sid, out, ref)
    return [engine.finished[sid] for sid in sids]


def main():
    banner("Full Engine - mini-vLLM")
    mc = ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=len(CharTokenizer()))   # vocab 必须与 tokenizer 一致
    lm = TinyLM(mc)
    SYSTEM = "You are a helpful assistant. "
    prompts = [SYSTEM + q for q in ["Tell me a joke.", "What is 2+2?", "Write a haiku.", "Translate hi.", "Sing a song."]]

    print("\n[1] 5 条共享 system prompt 的请求 (block=8, 每步 token 预算 48, 分块 prefill + 前缀缓存)")
    cfg = EngineConfig(model=mc, block_size=8, num_blocks=64, max_batch_seqs=4, max_batch_tokens=48)
    engine = Engine(cfg, lm)
    seqs = run_and_check(engine, prompts, max_new=20, verbose=True)
    s = engine.report_stats()
    need = sum(q.num_tokens - 1 for q in seqs)           # 每条序列最后采出的那个 token 不需要再算 KV
    kv("需要 KV 的 token 总数", need)
    kv("真正做了前向的 token", s["tokens_computed"])
    kv("前缀命中而跳过的 token", s["prefix_hit_tokens"])
    assert s["tokens_computed"] + s["prefix_hit_tokens"] == need and s["prefix_hit_tokens"] > 0
    print("  ✓ 输出与 generate_greedy 逐 token 相同; 命中 + 实算 = 总数 (省下的算力是真的)")

    print("\n[2] 同样 5 条再提交一次: 上一轮的 block 已全部释放, 但内容还在 → prompt 的完整 block 全部命中, 只剩尾巴 + decode 要算")
    before = s["tokens_computed"]
    run_and_check(engine, prompts, max_new=20)
    s2 = engine.report_stats()
    kv("第二轮实算 token (第一轮)", f"{s2['tokens_computed'] - before} ({before})")
    assert s2["tokens_computed"] - before < before
    assert s2["pool"]["used"] == 0

    print("\n[3] KV pool 只有 9 个 block (72 token), 单条请求就要 ~8 个 → 必然抢占")
    for chunked in (False, True):
        tiny = Engine(EngineConfig(model=mc, block_size=8, num_blocks=9, max_batch_seqs=4,
                                   max_batch_tokens=48, chunked_prefill=chunked), lm)
        run_and_check(tiny, prompts, max_new=20)
        st = tiny.report_stats()
        kv(f"chunked={chunked!s:<5} step / 抢占 / 实算 token", f"{st['steps']} / {st['preempt']} / {st['tokens_computed']}")
        assert st["preempt"] > 0 and st["pool"]["used"] == 0
    print("  ✓ 无活锁; 被抢占的序列重算后输出仍逐 token 相同, 长度不超 max_new")

    print("\n[4] 同一个 batch, 每条请求自带采样参数 (m10)")
    e4 = Engine(cfg, lm)
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.8, top_k=10),
              SamplingParams(temperature=1.0, top_p=0.9)]
    sids = [e4.add_request("same prompt", sp, max_new=12) for sp in params]
    while e4.has_unfinished():
        e4.step()
    for sid, sp in zip(sids, params):
        print(f"  T={sp.temperature:<4} top_k={sp.top_k!s:<5} top_p={sp.top_p!s:<5} → {e4.tok.decode(e4.finished[sid].output_ids)!r}")
    greedy_ref = lm.generate_greedy(np.array(e4.finished[sids[0]].prompt_ids), 12)[-12:]
    assert e4.finished[sids[0]].output_ids == [int(t) for t in greedy_ref][:len(e4.finished[sids[0]].output_ids)]

    print("\n[5] fuzz: 30 组随机 (block_size / pool / 预算 / 分块 / 前缀缓存) 配置, 含 id ≥ 256 的词表")
    lm_big = TinyLM(ModelConfig(vocab_size=400))
    rs = np.random.RandomState(1)
    n_req = n_preempt = 0
    for _ in range(30):
        bs, max_new = int(rs.choice([2, 4, 8])), int(rs.randint(1, 12))
        shared = list(rs.randint(3, 400, size=rs.randint(0, 20)))
        ps = [shared + list(rs.randint(3, 400, size=rs.randint(1, 15))) for _ in range(rs.randint(1, 7))]
        need_blocks = max(-(-(len(p) + max_new) // bs) for p in ps)
        e = Engine(EngineConfig(model=lm_big.cfg, block_size=bs, num_blocks=need_blocks + int(rs.randint(0, 6)),
                                max_batch_seqs=int(rs.randint(1, 5)), max_batch_tokens=int(rs.randint(1, 40)),
                                chunked_prefill=bool(rs.randint(2)), prefix_caching=bool(rs.randint(2))), lm_big)
        for _round in range(2):
            run_and_check(e, ps, max_new)
        bm = e.scheduler.bm
        assert bm.num_free_blocks() == bm.num_blocks and sum(bm.ref_count) == 0, "block 泄漏"
        n_req += 2 * len(ps)
        n_preempt += e.scheduler.preempt_count
    kv("请求数 / 其中触发的抢占", f"{n_req} / {n_preempt}")
    print("  ✓ 全部与 generate_greedy 逐 token 相同, 无活锁, 无 block 泄漏")


if __name__ == "__main__":
    main()
