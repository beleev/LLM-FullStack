"""按学习路径顺序跑完所有 demo; 任何一个 assert 失败都会让整体失败。"""
from __future__ import annotations

import importlib


DEMOS = [
    "llm_train.m01_gradient_accumulation.demo",
    "llm_train.m02_data_parallel.demo",
    "llm_train.m03_tensor_parallel.demo",
    "llm_train.m04_pipeline_parallel.demo",
    "llm_train.m05_zero_fsdp.demo",
    "llm_train.m06_mixed_precision.demo",
    "llm_train.m07_activation_checkpointing.demo",
    "llm_train.m08_checkpoint_resume.demo",
    "llm_train.m09_collectives.demo",
    "llm_train.m10_training_stability.demo",
    "llm_train.m11_expert_parallel.demo",
    "llm_train.m12_sequence_parallel.demo",
    "llm_train.m13_fp8_training.demo",
    "llm_train.m14_muon_optimizer.demo",
    "llm_train.m15_fp4_microscaling.demo",
    "llm_train.m16_ulysses_sequence_parallel.demo",
    "llm_train.full_loop.demo",
]


def main() -> None:
    for module_name in DEMOS:
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()

