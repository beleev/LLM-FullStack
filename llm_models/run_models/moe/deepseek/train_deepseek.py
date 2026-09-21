#!/usr/bin/env python
"""
DeepSeek-V3 训练示例 — 重点演示 aux-loss-free 负载均衡

同一个种子训两遍 (router 故意初始化得偏心), 唯一区别是每步之后调不调
    model.update_routing_bias(routing_info, gamma)      # bias_i += γ·sign(mean_load − load_i)
aux_loss_weight = 0: 均衡完全靠 bias, LM loss 的梯度不被任何辅助 loss 打扰 (aux_loss 仅作监控打印)。

断言: ① 初始 lm_loss ≈ ln V  ② loss 下降  ③ 开 bias 的负载变异系数 CV 明显更低  ④ 两次的 lm_loss 几乎一样。
数据: 固定的一个随机 batch —— "loss 下降" 只说明模型能背下这个 batch。
"""

import math

import torch

from llm_models.models.moe.deepseekV3 import DeepSeekV3
from llm_models.training import DecoderOnlyDataGenerator, MoELMLoss, Trainer, TrainingConfig

VOCAB, STEPS = 1000, 100


class _KeepRouting(MoELMLoss):
    """Trainer 不把模型输出交还给调用者, 这里顺手留一份 routing_info 给 bias 更新用。"""

    def compute(self, model_output, labels, **kwargs):
        self.routing_info = model_output[1]
        return super().compute(model_output, labels, **kwargs)


def train(gamma: float):
    config = TrainingConfig(
        learning_rate=3e-4, batch_size=4, seq_len=32, num_steps=STEPS,
        warmup_steps=5, aux_loss_weight=0.0, log_interval=25, seed=42,
    )
    torch.manual_seed(config.seed)
    model = DeepSeekV3(
        vocab_size=VOCAB, d_model=128, n_heads=4, num_layers=2, max_len=64,
        num_shared_experts=1, num_routed_experts=8, top_k=2,
        latent_dim=32, qk_rope_head_dim=16, dropout=0.0,
    )
    with torch.no_grad():  # 偏心的 router: 专家 0/1 的打分被放大 4 倍, 模拟路由开始坍塌
        for layer in model.layers:
            layer.moe.router.weight[:2] *= 4

    data_gen = DecoderOnlyDataGenerator(vocab_size=VOCAB, batch_size=config.batch_size, seq_len=config.seq_len)
    loss_fn = _KeepRouting(aux_loss_weight=config.aux_loss_weight)
    trainer = Trainer(model, config, data_gen, loss_fn)

    lm_losses, cvs = [], []
    for step in range(1, STEPS + 1):
        metrics = trainer.train_step()
        # optimizer.step() 之后、下一次 forward 之前更新 bias; gamma=0 时只统计 load
        load = model.update_routing_bias(loss_fn.routing_info, gamma=gamma)  # [L, E]
        cv = (load.std(dim=1) / load.mean(dim=1)).mean().item()              # 变异系数, 0 = 完全均衡
        lm_losses.append(metrics["lm_loss"].item())
        cvs.append(cv)
        if step == 1 or step % config.log_interval == 0:
            print(f"  step {step:>3d} | lm_loss {lm_losses[-1]:.4f} | aux(监控) {metrics['aux_loss'].item():.3f} "
                  f"| load CV {cv:.3f} | layer0 load {load[0].int().tolist()}")
    return model, lm_losses, sum(cvs[-20:]) / 20


def main():
    print("--- 不更新 bias (gamma=0) ---")
    _, loss_off, cv_off = train(gamma=0.0)
    print("--- aux-loss-free bias 更新 (gamma=1e-3) ---")
    model, loss_on, cv_on = train(gamma=1e-3)

    info = model.get_num_active_params()
    print(f"\n总参数 {info['total_params']:,} | 每 token 激活 {info['active_params']:,}")
    print(f"初始 lm_loss {loss_on[0]:.4f} (ln V = {math.log(VOCAB):.4f}) → 最终 {loss_on[-1]:.4f}")
    print(f"最后 20 步平均 load CV: 无 bias {cv_off:.3f}  vs  有 bias {cv_on:.3f}")
    print(f"layer0 routing_bias: {[round(b, 3) for b in model.layers[0].moe.routing_bias.tolist()]}")

    assert abs(loss_on[0] - math.log(VOCAB)) < 0.5, "初始 loss 应 ≈ ln V (init_weights 失效?)"
    assert loss_on[-1] < loss_on[0] - 1.0, "loss 未下降"
    assert cv_on < 0.7 * cv_off, "bias 更新没有让负载更均衡"
    assert abs(loss_on[-1] - loss_off[-1]) < 0.1, "bias 只改路由选择, 不应明显影响 LM loss"


if __name__ == "__main__":
    main()
