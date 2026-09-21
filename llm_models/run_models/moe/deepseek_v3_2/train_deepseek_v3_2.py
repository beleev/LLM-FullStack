#!/usr/bin/env python
"""
DeepSeek-V3.2 (DSA) 训练示例 — Lightning Indexer 到底是怎么被训练的

top-k 不可导, 所以只用 LM loss 时 indexer 的梯度是 None (阶段 0 先验证这一点)。
V3.2 的做法是给 indexer 单独一个对齐 loss:  index_loss = KL( 主注意力分布 ‖ softmax(indexer 分数) )
按论文的顺序分三段 (同一个固定 batch):
    A. 稠密预训练:   dense_warmup=True, 只训主模型 —— 让注意力先长出结构 (对应 V3.1 checkpoint)
    B. indexer 预热: 冻结主模型, 只用 index_loss 训 indexer → KL 下降, top-k 召回率上升
    C. 稀疏训练:     dense_warmup=False, 注意力真的只看 top-k; total = LM + aux + λ·index_loss

召回率 = indexer 选的 top-k 与 "稠密注意力真正最大的 k 个 key" 的重合比例 (只统计可见 key 多于 k 的行)。
数据是固定的一个随机 batch: loss 下降只说明能背下它。
"""

import math

import torch

from llm_models.models.moe.deepseekV3 import DeepSeekV3_2
from llm_models.training import DecoderOnlyDataGenerator, MoELMLoss

VOCAB, SEQ, TOPK = 1000, 32, 8


def topk_recall(model) -> float:
    """各层平均的 indexer top-k 召回率; 需在 dense_warmup=True 的 forward 之后调用。"""
    recalls = []
    for layer in model.layers:
        scores, target = layer.attn.last_index_scores, layer.attn.last_attn_target  # [B, T, S]
        rows = slice(TOPK, None)  # 第 t 行可见 t+1 个 key; 只有 > k 个时 top-k 才有得选
        pred = torch.zeros_like(target).scatter_(-1, scores.topk(TOPK, dim=-1).indices, 1.0)
        true = torch.zeros_like(target).scatter_(-1, target.topk(TOPK, dim=-1).indices, 1.0)
        recalls.append(((pred * true).sum(-1) / TOPK)[:, rows].mean().item())
    return sum(recalls) / len(recalls)


def main():
    torch.manual_seed(42)
    model = DeepSeekV3_2(
        vocab_size=VOCAB, d_model=128, n_heads=4, num_layers=2, max_len=64,
        num_routed_experts=4, num_shared_experts=1, top_k=2,
        latent_dim=32, qk_rope_head_dim=16, dropout=0.0,
        sparse_top_k=TOPK, indexer_heads=2, indexer_head_dim=32,
    )
    batch = DecoderOnlyDataGenerator(vocab_size=VOCAB, batch_size=4, seq_len=SEQ).generate_batch()
    idx, labels = batch["idx"], batch["labels"]
    indexer_params = [p for n, p in model.named_parameters() if "indexer" in n]
    main_params = [p for n, p in model.named_parameters() if "indexer" not in n]

    def step(loss_fn, opt):
        out = loss_fn.compute(model(idx), labels)
        model.zero_grad()
        out["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return {k: v.item() for k, v in out.items()}

    # ---- 0) 只有 LM loss 时, indexer 拿不到梯度 ----
    out = MoELMLoss(index_loss_weight=0.0).compute(model(idx), labels)
    lm_grads = torch.autograd.grad(out["lm_loss"], indexer_params, allow_unused=True)
    assert all(g is None for g in lm_grads), "top-k 不可导, LM loss 不应给 indexer 梯度"
    first_lm = out["lm_loss"].item()
    print(f"初始 lm_loss {first_lm:.4f} (ln V = {math.log(VOCAB):.4f}) | LM loss 对 indexer 的梯度: 全为 None")
    assert abs(first_lm - math.log(VOCAB)) < 0.5

    # ---- A) 稠密预训练: 只更新主模型 ----
    model.set_dense_warmup(True)
    opt = torch.optim.AdamW(main_params, lr=1e-3)
    for s in range(60):
        m = step(MoELMLoss(aux_loss_weight=0.01, index_loss_weight=0.0), opt)
    print(f"[A 稠密预训练 60 步]  lm_loss {first_lm:.4f} → {m['lm_loss']:.4f}")
    assert m["lm_loss"] < first_lm - 1.0

    # ---- B) indexer 预热: 冻结主模型, 只用 KL 训 indexer ----
    opt = torch.optim.AdamW(indexer_params, lr=3e-3)
    only_index = MoELMLoss(aux_loss_weight=0.0, index_loss_weight=1.0)
    for s in range(80):
        out = only_index.compute(model(idx), labels)
        model.zero_grad()
        out["index_loss"].backward()  # 输入和目标都 detach → 梯度只落在 indexer 上
        if s == 0:
            kl0, recall0 = out["index_loss"].item(), topk_recall(model)
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in indexer_params)
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in main_params)
        opt.step()
    kl1, recall1 = out["index_loss"].item(), topk_recall(model)
    print(f"[B indexer 预热 80 步] KL {kl0:.4f} → {kl1:.4f} | top-{TOPK} 召回率 {recall0:.3f} → {recall1:.3f}"
          f" (随机乱选的期望 ≈ {sum(TOPK / n for n in range(TOPK + 1, SEQ + 1)) / (SEQ - TOPK):.2f})")
    assert kl1 < 0.7 * kl0, "indexer 对齐 KL 未下降"
    assert recall1 > recall0 + 0.05, "indexer top-k 召回率未提升"

    # ---- C) 稀疏训练: 注意力只看 top-k; LM + aux + λ·index_loss 一起训 ----
    model.set_dense_warmup(False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    joint = MoELMLoss(aux_loss_weight=0.01, index_loss_weight=1.0)
    for s in range(40):
        m = step(joint, opt)
        if s == 0:
            m0 = m
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in indexer_params)
    print(f"[C 稀疏训练 40 步]    lm_loss {m0['lm_loss']:.4f} → {m['lm_loss']:.4f} | "
          f"index_loss {m0['index_loss']:.4f} → {m['index_loss']:.4f} | aux {m['aux_loss']:.3f}")
    assert m["lm_loss"] < m0["lm_loss"], "切到稀疏注意力后 LM loss 应继续下降"


if __name__ == "__main__":
    main()
