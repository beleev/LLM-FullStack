#!/usr/bin/env python
"""
知识蒸馏示例: 4 层 teacher → 2 层 student
==========================================

教学目标:
    - 蒸馏三件套: 硬标签 CE + 软标签 KL + 温度 T (乘 T² 补偿)
    - 直观看到温度把 teacher 分布"压平", 暗知识 (相对排序) 被放大
    - 学生在 CE+KD 联合监督下学会 teacher 记忆的任务

流程:
    1. teacher (4 层) 在固定 batch 上训练至记忆 (loss 大幅下降)
    2. student (2 层, ~1/3 参数) 用 DistillLoss 模仿 teacher + 硬标签
    3. 验证: student 的 CE 与 KD 都应显著下降

运行:
    python -m llm_finetune.run_finetune.distill.train_distill
"""

import torch

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import DecoderOnlyDataGenerator

from llm_finetune import DistillLoss, soften_demo


def make_model(num_layers: int, d_model: int, vocab_size: int) -> LLaMA:
    return LLaMA(
        vocab_size=vocab_size, d_model=d_model, n_heads=4, num_kv_heads=2,
        num_layers=num_layers, max_len=64, dropout=0.0,
    )


def main() -> None:
    torch.manual_seed(42)
    vocab_size = 200

    # ---- 0) 固定一个 batch (教学惯例: 可记忆 → loss 必然可降) ----
    gen = DecoderOnlyDataGenerator(vocab_size=vocab_size, batch_size=4, seq_len=24)
    batch = gen.generate_batch()
    idx, labels = batch["idx"], batch["labels"]

    # ---- 1) 训练 teacher 至记忆 ----
    teacher = make_model(num_layers=4, d_model=256, vocab_size=vocab_size)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=3e-4)
    ce = torch.nn.CrossEntropyLoss()
    for step in range(150):
        loss = ce(teacher(idx).reshape(-1, vocab_size), labels.reshape(-1))
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(idx)
        t_loss = float(ce(teacher_logits.reshape(-1, vocab_size), labels.reshape(-1)))
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"teacher (4 层, {n_t:,} 参数) 训练完成, CE = {t_loss:.4f}")

    # ---- 2) 温度的直观效果: 同一行 logits, T 越大分布越平 ----
    print("\nteacher 在样本 0 位置 0 的输出分布 (温度软化):")
    soften_demo(teacher_logits[0, 0])

    # ---- 3) 蒸馏 student ----
    student = make_model(num_layers=2, d_model=128, vocab_size=vocab_size)
    n_s = sum(p.numel() for p in student.parameters())
    print(f"\nstudent (2 层, {n_s:,} 参数, teacher 的 {n_s / n_t:.0%}) 开始蒸馏")

    distill = DistillLoss(temperature=2.0, alpha=0.3)
    opt_s = torch.optim.AdamW(student.parameters(), lr=3e-4)
    history = []
    for step in range(1, 201):
        out = distill.compute(student(idx), teacher_logits, labels)
        opt_s.zero_grad()
        out["total_loss"].backward()
        opt_s.step()
        history.append({k: float(v) for k, v in out.items()})
        if step % 50 == 0 or step == 1:
            h = history[-1]
            print(f"step {step:>3} | total {h['total_loss']:.3f} "
                  f"| ce {h['ce_loss']:.3f} | kd {h['kd_loss']:.3f}")

    first, last = history[0], history[-1]
    assert last["ce_loss"] < first["ce_loss"], "student CE 未下降"
    assert last["kd_loss"] < first["kd_loss"], "student KD 未下降"
    print(f"\n蒸馏通过: CE {first['ce_loss']:.2f}→{last['ce_loss']:.2f}, "
          f"KD {first['kd_loss']:.2f}→{last['kd_loss']:.2f}")
    print("R1 蒸馏小模型用的是同一思想的序列版: teacher 生成文本, student 做 SFT。")


if __name__ == "__main__":
    main()
