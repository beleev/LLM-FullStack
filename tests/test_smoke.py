"""冒烟测试: 把仓库里每一个可运行的教学脚本当子进程跑一遍, 退出码非 0 即失败。

脚本清单是扫目录动态发现的 —— 新增一个 mNN_xxx/demo.py 会自动被测到, 不需要维护列表。
各 demo 自己负责在末尾 assert 它声称的结论, 这里只负责"全都真的跑过"。

    pytest                # 快速集: 所有 demo.py / infer_*.py
    pytest -m slow        # 训练脚本 (train_*.py), 每个几秒到几十秒
    pytest -m ""          # 全部
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["llm_models", "llm_train", "llm_finetune", "llm_infer", "llm_agent"]


def _modules(pattern):
    """'llm_infer/m01_kv_cache/demo.py' -> 'llm_infer.m01_kv_cache.demo'"""
    found = []
    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob(pattern)):
            if "__pycache__" not in path.parts:
                found.append(".".join(path.relative_to(ROOT).with_suffix("").parts))
    return found


def _run(args, cwd=ROOT, timeout=300):
    proc = subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    # 失败时把输出尾部带进断言信息, CI 日志里直接能看到原因
    assert proc.returncode == 0, f"{' '.join(args)} 失败:\n{(proc.stdout + proc.stderr)[-2000:]}"


@pytest.mark.parametrize("module", _modules("demo.py") + _modules("infer_*.py"))
def test_demo(module):
    _run(["-m", module])


@pytest.mark.slow
@pytest.mark.parametrize("module", _modules("train_*.py"))
def test_train_script(module):
    _run(["-m", module], timeout=600)


def test_main_smoke():
    _run(["main.py"])


def test_basic_gradcheck():
    # llm_basic 不是包 (脚本直接在目录里跑), 所以切到它的目录执行
    _run(["gradcheck.py"], cwd=ROOT / "llm_basic")
