#!/usr/bin/env python
"""
Mixtral 训练示例: LM loss + Switch 风格负载均衡 aux loss

先用手工构造的路由验证 aux loss 的两个极端值 (E 个专家, 每 token 选 K 个):
    完全均衡 → aux = K   (不是 1! f_i 按 K 次选择计数, Σ f_i = K)
    完全坍塌 → aux = E
再训练 50 步。数据是固定的一个随机 batch: loss 下降只说明模型能背下它。
"""

import math

import torch

from llm_models.models.moe.mixtral import Mixtral
from llm_models.training import DecoderOnlyDataGenerator, MoELMLoss, Trainer, TrainingConfig


def check_aux_loss_extremes(E: int = 8, K: int = 2, N: int = 64):
    aux = MoELMLoss._compute_load_balancing_loss
    t = torch.arange(N)

    # 均衡: 概率均匀, token t 选专家 (t, t+1) mod E → 每个专家恰好 N·K/E 次
    balanced = {
        "routing_probs": torch.full((N, E), 1.0 / E),
        "selected_experts": torch.stack([(t + j) % E for j in range(K)], dim=1),
    }
    # 坍塌: 所有 token 都把概率压在专家 0..K-1 上并选中它们
    probs = torch.zeros(N, E)
    probs[:, :K] = 1.0 / K
    collapsed = {"routing_probs": probs, "selected_experts": torch.arange(K).expand(N, K)}

    a_bal, a_col = aux([balanced]).item(), aux([collapsed]).item()
    print(f"aux loss 极值 (E={E}, K={K}): 均衡 = {a_bal:.4f} (应为 K), 坍塌 = {a_col:.4f} (应为 E)")
    assert abs(a_bal - K) < 1e-5 and abs(a_col - E) < 1e-5

    # 没有 MoE 层 (routing_info 为空) 时 aux 必须是与 logits 同设备的 0, total == lm
    logits = torch.zeros(1, 4, 10)
    out = MoELMLoss().compute((logits, []), torch.zeros(1, 4, dtype=torch.long))
    assert out["aux_loss"].device == logits.device and out["total_loss"] == out["lm_loss"]


def main():
    check_aux_loss_extremes()

    cfg = TrainingConfig(
        learning_rate=3e-4, batch_size=2, seq_len=32,
        num_steps=50, warmup_steps=5, aux_loss_weight=0.01,
        log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    vocab_size = 1000
    model = Mixtral(
        vocab_size=vocab_size, d_model=128, n_heads=4, num_kv_heads=2,
        num_layers=2, num_experts=4, top_k=2, max_len=64,
    )
    print(f"Mixtral Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DecoderOnlyDataGenerator(
        vocab_size=vocab_size, batch_size=cfg.batch_size, seq_len=cfg.seq_len,
    )
    metrics = Trainer(model, cfg, data_gen, MoELMLoss(aux_loss_weight=cfg.aux_loss_weight)).train()

    first, last = metrics[0], metrics[-1]
    print(f"lm_loss {first['lm_loss']:.4f} (ln V = {math.log(vocab_size):.4f}) → {last['lm_loss']:.4f} | "
          f"aux_loss {first['aux_loss']:.4f} → {last['aux_loss']:.4f} (均衡值 K = 2, 坍塌值 E = 4)")
    assert abs(first["lm_loss"] - math.log(vocab_size)) < 0.5, "初始 loss 应 ≈ ln V"
    assert last["lm_loss"] < first["lm_loss"] - 1.0, "loss 未下降"
    assert abs(first["aux_loss"] - 2.0) < 0.1, "小初始化下路由近似均匀, aux 应 ≈ K"


if __name__ == "__main__":
    main()
