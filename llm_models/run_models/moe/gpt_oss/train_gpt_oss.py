#!/usr/bin/env python
"""
GPT-OSS 训练示例 — LM loss + 负载均衡 aux loss, 用 assert 验证:
    1. 初始 lm_loss ≈ ln V (init_weights 生效; sink 初始 0 不破坏它)
    2. lm_loss 明显下降。数据是 **固定的一个随机 batch**: 下降 = 模型背下了它, 只证明通路正确
    3. sink logit 真的在学: 反传后梯度非零, 训练后值离开了初始的 0
    4. 打印每层专家负载 (均衡时每个专家 = K/E)
"""

import math

import torch

from llm_models.models.moe.gpt_oss import GPTOSSMini
from llm_models.training import DecoderOnlyDataGenerator, MoELMLoss, Trainer, TrainingConfig


def main():
    cfg = TrainingConfig(
        learning_rate=1e-3, batch_size=2, seq_len=32,
        num_steps=60, warmup_steps=5, aux_loss_weight=0.01,
        log_interval=10, seed=42,
    )
    torch.manual_seed(cfg.seed)

    V, E, K = 1000, 4, 2
    model = GPTOSSMini(vocab_size=V, d_model=128, n_heads=4, num_kv_heads=2, num_layers=4,
                       num_experts=E, top_k=K, max_len=64, window_size=8)
    print(f"GPT-OSS Mini | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    data_gen = DecoderOnlyDataGenerator(vocab_size=V, batch_size=cfg.batch_size, seq_len=cfg.seq_len)
    sinks = lambda: torch.stack([layer.attn.sink.detach().clone() for layer in model.layers])  # [L, H]
    assert (sinks() == 0).all()
    metrics = Trainer(model, cfg, data_gen, MoELMLoss(aux_loss_weight=cfg.aux_loss_weight)).train()

    first, last = metrics[0], metrics[-1]
    ln_v = math.log(V)
    print(f"lm_loss {first['lm_loss']:.4f} (ln V = {ln_v:.4f}) → {last['lm_loss']:.4f} | "
          f"aux_loss {first['aux_loss']:.4f} → {last['aux_loss']:.4f} (均衡值 K = {K}, 坍塌值 E = {E})")
    assert abs(first["lm_loss"] - ln_v) < 0.5, "初始 loss 应 ≈ ln V"
    assert last["lm_loss"] < 0.5 * first["lm_loss"], "loss 未明显下降"

    # sink: 值已离开 0, 且再反传一次梯度非零 (Trainer 每步后把 grad 置 None, 所以这里手动反传)
    batch = data_gen.generate_batch()
    logits, routing = model(batch["idx"])
    MoELMLoss().compute((logits, routing), batch["labels"])["total_loss"].backward()
    grads = torch.stack([layer.attn.sink.grad for layer in model.layers])                     # [L, H]
    assert (grads != 0).all() and (sinks() != 0).all(), "sink logit 没有收到梯度"
    for i, s in enumerate(sinks()):
        kind = "SWA " if model.is_swa(i) else "full"
        print(f"  Layer {i} ({kind}) sink logits: {[round(v, 4) for v in s.tolist()]}")
    print(f"  sink 梯度 |g| 最大 {grads.abs().max():.2e}, 最小 {grads.abs().min():.2e} (全部非零)")

    # 专家负载: 每个专家被选中的 token 比例 (K 个选择, 每行之和 = K)
    for i, info in enumerate(routing):
        load = torch.bincount(info["selected_experts"].flatten(), minlength=E).float()
        load = load / info["selected_experts"].size(0)
        assert abs(load.sum().item() - K) < 1e-5
        print(f"  Layer {i} 专家负载: {[round(v, 2) for v in load.tolist()]}  (均衡 = {K / E:.2f})")


if __name__ == "__main__":
    main()
