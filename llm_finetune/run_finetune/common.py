"""run_finetune 下所有脚本共用的三件事: 小模型、通用 Trainer 的一行封装、SFT 热身。"""

from typing import Dict, List

import torch
import torch.nn as nn

from llm_models.models.language_models.llama import LLaMA
from llm_models.training import Trainer, TrainingConfig
from llm_models.training.data import SyntheticDataGenerator
from llm_models.training.loss import LossComputer

from llm_finetune.data import InstructionDataGenerator, SeqTask
from llm_finetune.methods.sft import SFTLoss

# d_model=64 的模型每个 matmul 只有几微秒, 多线程的同步开销反而比计算大: 单线程实测快约 2 倍
torch.set_num_threads(1)


def make_model(task: SeqTask, d_model: int = 64, num_layers: int = 2) -> LLaMA:
    return LLaMA(vocab_size=task.vocab_size, d_model=d_model, n_heads=4, num_kv_heads=2,
                 num_layers=num_layers, max_len=32)


def fit(model: nn.Module, data: SyntheticDataGenerator, loss: LossComputer, steps: int,
        lr: float, log_interval: int = 100) -> List[Dict[str, float]]:
    """通用 Trainer 跑 steps 步。model 可以是裸 LM, 也可以是 PairwiseForward / TeacherStudent 这类包装。"""
    cfg = TrainingConfig(learning_rate=lr, num_steps=steps, warmup_steps=max(1, steps // 20),
                         log_interval=log_interval)
    return Trainer(model, cfg, data, loss).train()


def sft_warmup(model: nn.Module, task: SeqTask, steps: int, lr: float = 3e-3) -> float:
    """偏好对齐 / RL 都从 SFT 过的模型出发 (真实流水线也是 SFT → DPO / GRPO)。返回留出集 exact-match。"""
    fit(model, InstructionDataGenerator(task), SFTLoss(), steps, lr, log_interval=steps)
    return task.exact_match(model)


def pretrained_base(steps: int = 150) -> LLaMA:
    """PEFT 脚本共用的 "预训练基座": 已经会 copy 任务 (留出集 EM = 1), 接下来要被适配到 sort。"""
    task = SeqTask("copy")
    base = make_model(task)
    em = sft_warmup(base, task, steps)
    assert em > 0.95, f"基座没训好: copy EM {em:.3f}"
    return base


@torch.no_grad()
def preference_accuracy(policy: nn.Module, task: SeqTask, n: int = 256, seed: int = 4321) -> Dict[str, float]:
    """留出集偏好对上, policy 给正确回复的概率高于损坏回复的比例 (直接比 log π(y|x), 不依赖任何 ref)。"""
    from llm_finetune.data import PreferenceDataGenerator
    from llm_finetune.methods.dpo import compute_sequence_logprobs

    state = torch.get_rng_state()
    torch.manual_seed(seed)                                   # 固定种子: 每次评的是同一批留出偏好对
    batch = PreferenceDataGenerator(task, n, split="test").generate_batch()
    torch.set_rng_state(state)
    was_training = policy.training
    policy.eval()
    logp = {k: compute_sequence_logprobs(policy(batch[f"{k}_input_ids"]), batch["labels"][k])
            for k in ("chosen", "rejected")}
    policy.train(was_training)
    return {"accuracy": float((logp["chosen"] > logp["rejected"]).float().mean()),
            "logp_chosen": float(logp["chosen"].mean()), "logp_rejected": float(logp["rejected"].mean())}
