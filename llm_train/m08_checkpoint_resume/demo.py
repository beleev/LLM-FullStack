"""
M08 — Checkpoint / Resume

可靠训练不只保存模型参数, 还要保存:
    - optimizer states
    - step
    - data cursor / sampler state
    - RNG state

本 demo 验证: uninterrupted 训练 5 步 == 训练 3 步保存后恢复再训练 2 步。
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from llm_train.core import LinearModel, MomentumSGD, ToyDataStream, banner, kv


def train_steps(model, optim, data, steps: int) -> None:
    for _ in range(steps):
        x, y = data.next_batch()
        _, grads = model.loss_and_grads(x, y)
        optim.step(model.params(), grads)


def save_checkpoint(path: Path, model, optim, data, step: int, rng_state) -> None:
    payload = {
        "model": {k: v.copy() for k, v in model.params().items()},
        "optim": optim.state_dict(),
        "data": data.state_dict(),
        "step": step,
        "rng_state": rng_state,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: Path, model, optim, data):
    with path.open("rb") as f:
        payload = pickle.load(f)
    model.load_params(payload["model"])
    optim.load_state_dict(payload["optim"])
    data.load_state_dict(payload["data"])
    return payload["step"], payload["rng_state"]


def main() -> None:
    banner("M08 - Checkpoint / Resume")

    # Reference run: 5 continuous steps.
    ref_model = LinearModel.init(4, 2, seed=8)
    ref_opt = MomentumSGD(ref_model.params(), lr=0.05, momentum=0.8)
    ref_data = ToyDataStream(4, 2, batch_size=6, seed=9)
    train_steps(ref_model, ref_opt, ref_data, steps=5)

    # Interrupted run: 3 steps, save, restore, 2 more steps.
    model = LinearModel.init(4, 2, seed=8)
    opt = MomentumSGD(model.params(), lr=0.05, momentum=0.8)
    data = ToyDataStream(4, 2, batch_size=6, seed=9)
    train_steps(model, opt, data, steps=3)

    with tempfile.TemporaryDirectory(prefix="llm_train_ckpt_") as tmp:
        path = Path(tmp) / "step_0003.pkl"
        save_checkpoint(path, model, opt, data, step=3, rng_state=np.random.get_state())

        restored = LinearModel.init(4, 2, seed=999)
        restored_opt = MomentumSGD(restored.params(), lr=0.01, momentum=0.0)
        restored_data = ToyDataStream(4, 2, batch_size=6, seed=9)
        step, _ = load_checkpoint(path, restored, restored_opt, restored_data)
        train_steps(restored, restored_opt, restored_data, steps=2)

        w_diff = np.max(np.abs(ref_model.W - restored.W))
        b_diff = np.max(np.abs(ref_model.b - restored.b))
        kv("checkpoint path", str(path))
        kv("restored step", step)
        kv("restored data cursor", restored_data.cursor)
        kv("max W diff vs uninterrupted", f"{w_diff:.2e}")
        kv("max b diff vs uninterrupted", f"{b_diff:.2e}")

    assert w_diff < 1e-7 and b_diff < 1e-7
    print("\n  OK: 断点恢复必须覆盖参数、优化器、数据进度和随机状态。")


if __name__ == "__main__":
    main()

