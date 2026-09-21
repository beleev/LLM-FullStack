#!/usr/bin/env python
"""
GRPO / DAPO / Dr.GRPO / GSPO: 同一个 SFT 起点、同样的 prompt 流和步数, 奖励 = 任务 verifier (排序全对才得 1 分)。
先用几个确定性的小实验把每个变体改动的那一项单独量出来, 再看训练结果。

    python -m llm_finetune.run_finetune.grpo.train_grpo
"""

import copy

import torch
import torch.nn.functional as F

from llm_finetune import (
    GRPOTrainer, PromptDataGenerator, SeqTask, aggregate, completion_logprobs, group_advantages, make_config,
)
from llm_finetune.run_finetune.common import make_model, sft_warmup

SFT_STEPS, RL_STEPS, PROMPTS_PER_STEP, LR = 150, 60, 32, 3e-4


def diagnostics(policy, task) -> None:
    print("---- 变体各改了什么 (确定性小实验) ----")
    # (a) ÷std: 8 条里对 1 条 (难题) vs 对 4 条 (中等题), 看每道题拿到的总权重 mean|A|
    hard, medium = torch.tensor([1.] + [0.] * 7), torch.tensor([1.] * 4 + [0.] * 4)
    weight = {norm: [float(group_advantages(r, 8, norm).abs().mean()) for r in (hard, medium)] for norm in (True, False)}
    print(f"(a) 题目权重 难题/中等题: GRPO (÷std) {weight[True][0] / weight[True][1]:.2f} | "
          f"Dr.GRPO (不÷std) {weight[False][0] / weight[False][1]:.2f}   ← ÷std 把没什么可学的极端题目重新放大")
    assert weight[True][0] / weight[True][1] > weight[False][0] / weight[False][1]

    # (b) 聚合方式: 一条 3-token 回复和一条 9-token 回复, 对 per-token loss 求导 = 每个 token 的权重
    mask = torch.zeros(2, 9)
    mask[0, :3], mask[1, :] = 1, 1
    ratio = {}
    for how in ("seq_mean", "token_mean", "fixed_len"):
        per_token = torch.zeros(2, 9, requires_grad=True)
        aggregate(per_token, mask, how).backward()
        ratio[how] = float(per_token.grad[0, 0] / per_token.grad[1, 0])
    print(f"(b) 短回复 token 权重 / 长回复 token 权重: GRPO seq_mean {ratio['seq_mean']:.1f} | DAPO token_mean "
          f"{ratio['token_mean']:.1f} | Dr.GRPO fixed_len {ratio['fixed_len']:.1f}   ← seq_mean 下答错的长回复每 token 受罚只有 1/3")
    assert ratio["seq_mean"] == 3.0 and ratio["token_mean"] == ratio["fixed_len"] == 1.0

    # (c) 温度一致性: T=0.5 采样 4000 个首 token, 经验分布应贴合 softmax(z/T) 而不是 softmax(z)
    T, prompt = 0.5, task.sample_prompts(1)
    first = policy.generate(prompt.repeat(4000, 1), 1, temperature=T)[:, -1]
    freq = torch.bincount(first, minlength=task.vocab_size) / 4000
    with torch.no_grad():
        z = policy(prompt)[0, -1]
    tv = {name: float((freq - F.softmax(z / t, -1)).abs().sum() / 2) for name, t in (("z/T", T), ("z", 1.0))}
    print(f"(c) T={T} 采样的经验分布与 softmax(z/T) 的 TV 距离 {tv['z/T']:.3f}, 与 softmax(z) 的 {tv['z']:.3f}   "
          f"← log-prob 必须带同一个 T, 否则 ρ 的分母不是行为策略")
    assert tv["z/T"] < tv["z"]
    policy.train()
    seqs = policy.generate(prompt.repeat(4, 1), 3, temperature=T).clone()
    assert policy.training, "generate() 结束后应恢复调用前的 train 状态 (否则采样一次就把模型永久留在 eval)"
    with torch.no_grad():
        assert not torch.allclose(completion_logprobs(policy, seqs, prompt.size(1), T),
                                  completion_logprobs(policy, seqs, prompt.size(1), 1.0))


def main() -> None:
    torch.manual_seed(0)
    task = SeqTask("sort")
    sft = make_model(task)
    em_sft = sft_warmup(sft, task, SFT_STEPS)                 # R1 式流水线: 先 SFT 到 "半会", 再 RL
    pass1_sft = task.exact_match(sft, n=512, temperature=1.0)  # 按策略采样的留出集正确率: RL 直接优化的就是它
    diagnostics(sft, task)

    print(f"\n---- 训练: 每步 {PROMPTS_PER_STEP} 个 prompt × G=8, 共 {RL_STEPS} 步; SFT 起点留出集: 贪心 EM {em_sft:.3f}, 采样 pass@1 {pass1_sft:.3f} ----")
    results = {}
    for variant in ("grpo", "dapo", "dr_grpo", "gspo"):
        torch.manual_seed(1)
        policy = copy.deepcopy(sft)
        trainer = GRPOTrainer(policy, task.verify, make_config(variant, lr=LR, max_new=task.response_len + 2))
        prompts = PromptDataGenerator(task, PROMPTS_PER_STEP)
        hist = [trainer.step(prompts.generate_batch()["prompts"]) for _ in range(RL_STEPS)]
        hist = [h for h in hist if "skipped" not in h]
        avg = lambda k: sum(h[k] for h in hist) / len(hist)
        results[variant] = dict(em=task.exact_match(policy), pass1=task.exact_match(policy, n=512, temperature=1.0), reward_first=hist[0]["reward"], reward_last=hist[-1]["reward"],
                                **{k: avg(k) for k in ("zero_adv_frac", "used_zero_adv_frac", "clip_frac",
                                                       "ratio_dev_epoch1", "ratio_dev_last", "log_ratio_std", "length")})

    print(f"{'':<9}{'贪心EM':>8}{'pass@1':>8}{'训练奖励 首→末':>16}{'A≡0 的 prompt':>14}{'进 loss 的 A=0':>14}"
          f"{'|ρ−1| ep1':>11}{'|ρ−1| ep2':>11}{'std log ρ':>11}{'裁剪比例':>9}")
    for v, r in results.items():
        print(f"{v:<9}{r['em']:>9.3f}{r['pass1']:>8.3f}{r['reward_first']:>11.3f}→{r['reward_last']:.3f}{r['zero_adv_frac']:>16.1%}"
              f"{r['used_zero_adv_frac']:>16.1%}{r['ratio_dev_epoch1']:>11.3f}{r['ratio_dev_last']:>11.3f}"
              f"{r['log_ratio_std']:>11.4f}{r['clip_frac']:>10.2%}")

    g, dapo, gspo = results["grpo"], results["dapo"], results["gspo"]
    for v, r in results.items():
        assert r["ratio_dev_epoch1"] == 0 and r["ratio_dev_last"] > 0, f"{v}: 第 1 个 epoch ρ≡1, 第 2 个 epoch 起 ρ≠1"
        assert r["pass1"] > pass1_sft + 0.04, f"{v}: 留出集 pass@1 应高于 SFT 起点 ({r['pass1']:.3f} vs {pass1_sft:.3f})"
        assert r["em"] > em_sft - 0.15, f"{v}: 贪心 EM 不应崩 ({r['em']:.3f} vs {em_sft:.3f})"
    assert g["used_zero_adv_frac"] > 0 == dapo["used_zero_adv_frac"], "动态采样应把 A≡0 的组全部剔除"
    assert gspo["log_ratio_std"] < 0.5 * g["log_ratio_std"], "序列级 ρ 的波动应远小于 token 级"


if __name__ == "__main__":
    main()
