"""
full_engine demo — mini-vLLM 集成演示

5 条请求, 共享 system prompt, 演示:
    1) prefill 优先调度
    2) 前缀缓存命中节省 prefill token
    3) decode 阶段 continuous batching
    4) 不同 sampling 参数同 batch 跑
"""
from __future__ import annotations
from llm_infer.core.utils import banner, kv
from llm_infer.core import ModelConfig, CharTokenizer
from llm_infer.m10_sampling.samplers import SamplingParams
from llm_infer.full_engine.engine import Engine, EngineConfig


def main():
    banner("Full Engine - mini-vLLM (paged + cont-batch + prefix-cache + sampling)")

    # 关键: 模型 vocab_size 必须与 tokenizer 一致, 否则 sample 出的 id 解码越界
    vocab = len(CharTokenizer())
    cfg = EngineConfig(
        model=ModelConfig(d_model=64, d_mlp=128, n_layer=4, vocab_size=vocab, max_seq_len=512),
        block_size=8,
        num_blocks=64,
        max_batch_seqs=4,
        max_batch_tokens=128,
    )
    engine = Engine(cfg)

    SYSTEM = "You are a helpful assistant. "
    requests = [
        SYSTEM + "Tell me a joke.",
        SYSTEM + "What is 2+2?",
        SYSTEM + "Write a haiku.",
        SYSTEM + "Translate hi.",
        SYSTEM + "Sing a song.",
    ]

    print(f"\n[1] 提交 {len(requests)} 条请求 (共享 system prompt)")
    sp = SamplingParams(temperature=0.0)        # greedy
    for i, r in enumerate(requests):
        sid = engine.add_request(r, sampling=sp, max_new=20)
        print(f"  seq {sid}: {r!r}")

    # ---- 主循环 ------------------------------------------------- #
    print("\n[2] 跑 step 直到全部完成")
    step = 0
    while engine.has_unfinished():
        step += 1
        finished = engine.step()
        if finished:
            for sid, txt in finished:
                print(f"  step {step:>3}  ✓ seq {sid} done: {txt!r}")
        else:
            phase = "PREFILL" if engine.waiting else "DECODE"
            print(f"  step {step:>3}  [{phase}] running={len(engine.running)} waiting={len(engine.waiting)}")

    # ---- 统计 -------------------------------------------------- #
    banner("Stats")
    s = engine.report_stats()
    kv("总 step", s["steps"])
    kv("prefix block 命中次数", s["prefix_block_hits"])
    kv("节省的 prefill token", s["prefill_tokens_saved"])
    kv("最终 pool stats", s["pool"])
    print("\n  注意: 后 4 条请求都共享 SYSTEM 前缀, 第 2 条起 prefix cache 命中")
    print("        节省了大量重复 prefill 计算")

    # ---- 不同 sampling 参数同 batch ---------------------------- #
    banner("[3] 不同 sampling 参数同 batch")
    engine2 = Engine(cfg)
    out = engine2.generate(
        prompts=["short", "another", "test prompt"],
        sampling=SamplingParams(temperature=0.7, top_k=10),
        max_new=12,
    )
    for sid, txt in out.items():
        print(f"  seq {sid}: {txt!r}")
    print("\n  ✓ 不同请求可带不同 SamplingParams (本例统一传, engine 支持单条覆盖)")


if __name__ == "__main__":
    main()
