"""Run all llm_train demos in learning-path order."""
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
    "llm_train.full_loop.demo",
]


def main() -> None:
    for module_name in DEMOS:
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()

