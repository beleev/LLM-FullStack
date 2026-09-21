#!/usr/bin/env python
"""
LLaDA 训练示例 — 在 "置换链" 上训练掩码扩散 LM, 然后用填空准确率验证它真的学会了生成。

任务: 固定一个随机置换 π (P 个 token), 序列 x[i+1] = π(x[i])。任何一个已知 token 都决定整条序列,
      但只给一个 token 时, 1 步并行要同时猜对 π^1..π^15 → 正好能看出 "先定有把握的, 再靠它们推其余" 的价值。
      每步都抽新数据 + 新的遮蔽, 所以 loss 下降是真的学会了 π, 不是背住一个 batch。
为什么不用 Trainer: 加噪要发生在 forward 之前且每步重新随机, Trainer 的 (batch → model → loss) 管线放不下。
理论下界: 整条序列只有起点是随机的, −log p(x) / L = ln P / L。
读代码时盯住: forward_process 返回的 t 和 masked。
"""

import math
import time

import torch

from llm_models.models.language_models.llada import LLaDA, LLaDALoss, forward_process

P, L = 16, 16  # 数据 token 数 / 序列长度; 词表 = P + 1, 最后一个 id 是 [MASK]


def make_batch(perm: torch.Tensor, batch_size: int, generator: torch.Generator = None) -> torch.Tensor:
    """x[:, 0] 随机, x[:, i+1] = perm[x[:, i]] → [B, L]"""
    cols = [torch.randint(0, len(perm), (batch_size,), generator=generator)]
    for _ in range(L - 1):
        cols.append(perm[cols[-1]])
    return torch.stack(cols, dim=1)


@torch.no_grad()
def eval_loss(model: LLaDA, x: torch.Tensor, seed: int = 123) -> float:
    """固定数据 + 固定遮蔽随机数 → 训练前后可比的 ELBO 估计。"""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    noisy, masked, t = forward_process(x, model.mask_id, generator=g)
    return LLaDALoss().compute(model(noisy), x, masked=masked, t=t)["total_loss"].item()


def train(steps: int = 400, seed: int = 0, log_interval: int = 100):
    torch.manual_seed(seed)
    order = torch.randperm(P)
    perm = torch.empty(P, dtype=torch.long)
    perm[order] = order.roll(-1)  # 单个 P-环: 避免短环让任务退化成 "重复 3 个 token"
    model = LLaDA(vocab_size=P + 1, d_model=64, n_heads=4, num_kv_heads=2, num_layers=2, max_len=L)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = LLaDALoss()
    for step in range(1, steps + 1):
        model.train()
        x = make_batch(perm, 64)  # [B, L] 每步新数据
        noisy, masked, t = forward_process(x, model.mask_id)  # 每步新遮蔽
        loss = loss_fn.compute(model(noisy), x, masked=masked, t=t)["total_loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_interval == 0:
            print(f"step {step:4d} | loss {loss.item():.3f}")
    return model, perm


def infill_accuracy(model: LLaDA, x: torch.Tensor, known: slice, steps: int, remasking: str) -> float:
    """只保留 x[:, known], 其余全遮, 让采样器还原; 返回被遮位置的准确率。"""
    noisy = torch.full_like(x, model.mask_id)
    noisy[:, known] = x[:, known]
    out = model.sample(noisy, steps, remasking=remasking)
    hidden = noisy == model.mask_id
    return (out[hidden] == x[hidden]).float().mean().item()


def main():
    t0 = time.time()
    V = P + 1

    # ---- 初始 loss: 均匀猜测时 CE = ln V, 且 E[被遮个数 / t] = L → 1/t 加权后期望仍是 ln V ----
    # 实测会比 ln V 高 ~0.1: weight tying 让 [MASK] 位置的 h ≈ e_[MASK], 初始时最偏爱输出 [MASK] 自己
    # (logit ≈ ‖e‖·√D ≈ 1.3 → CE ≈ ln(V − 1 + e^1.3) ≈ 2.95)。
    torch.manual_seed(0)
    fresh = LLaDA(vocab_size=V, d_model=64, n_heads=4, num_kv_heads=2, num_layers=2, max_len=L)
    init_loss = eval_loss(fresh, torch.randint(0, P, (1024, L)))  # 未训练时数据内容无关紧要
    print(f"初始 loss {init_loss:.3f}  vs  ln V = {math.log(V):.3f}")
    assert abs(init_loss - math.log(V)) < 0.3, "初始 loss 应 ≈ ln V (1/t 加权后期望不变)"

    model, perm = train()
    held_out = make_batch(perm, 1024, torch.Generator().manual_seed(999))
    final_loss = eval_loss(model, held_out)
    floor = math.log(P) / L
    print(f"训练后 held-out loss {final_loss:.3f}  (理论下界 ln P / L = {floor:.3f})")
    assert final_loss < 0.5 * init_loss, "loss 未明显下降"
    assert final_loss > floor - 0.05, "ELBO 是 NLL 的上界, 不该低于下界"

    # ---- 采样器结构性检查: 无 [MASK] / prompt 不被改 / [MASK] 个数走线性日程 ----
    prompt = held_out[:, :1]
    out, history = model.generate(prompt, L - 1, steps=4, return_history=True)
    assert not (out == model.mask_id).any(), "输出里不应有 [MASK]"
    assert (out[:, :1] == prompt).all(), "prompt 不应被改动"
    counts = [int((h[0] == model.mask_id).sum()) for h in history]
    expect = [round((L - 1) * (1 - s / 4)) for s in range(1, 5)]
    print(f"每步剩余 [MASK] 数 {counts}  期望 {expect}")
    assert counts == expect

    # ---- 填空准确率: 三种方向 × 两种重遮策略 (chance = 1/P) ----
    print(f"\n填空准确率 (held-out 1024 条, chance = {1 / P:.3f})")
    cases = {"给首 token 续写": slice(0, 1), "给末 token 倒推": slice(L - 1, L), "给中间 token 两头填": slice(8, 9)}
    acc = {}
    for name, known in cases.items():
        one = infill_accuracy(model, held_out, known, 1, "low_confidence")
        low = infill_accuracy(model, held_out, known, L - 1, "low_confidence")
        rnd = infill_accuracy(model, held_out, known, L - 1, "random")
        acc[name] = (one, low, rnd)
        print(f"  {name:<12} | 1 步并行 {one:.3f} | {L - 1} 步 low-confidence {low:.3f} | {L - 1} 步 random {rnd:.3f}")
    for name, (one, low, rnd) in acc.items():
        assert low > 0.8, f"{name}: 准确率 {low:.3f} 应远高于 chance"
    # 单个 case 上 low-confidence 偶尔会因 "自信地早早定错一个" 而略输 (换种子可见), 所以比三种方向的均值
    mean_one, mean_low, mean_rnd = (sum(v[i] for v in acc.values()) / len(acc) for i in range(3))
    assert mean_low >= mean_rnd, "low-confidence 重遮应不差于 random 重遮"
    assert mean_low >= mean_one, "多步去噪应不差于 1 步并行"

    print(f"\nLLaDA 训练验证通过 ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
