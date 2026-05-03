"""
m14 demo — 用 FSM 约束生成合法 JSON

我们做一个迷你的 grammar:
    grammar = "{" key_value ("," key_value)* "}"
    key_value = '"' name '"' ":" value
    value = number | '"' name '"'
    name = [a-z]+
    number = [0-9]+

不依赖任何 grammar 库, 用一个手写的状态机演示思想。
任意 logits 喂进来, 都能保证输出严格合法。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set
import numpy as np

from llm_infer.core import CharTokenizer
from llm_infer.core.utils import banner, kv, softmax


class State(Enum):
    EXPECT_OPEN = "expect_open"          # 期待 {
    EXPECT_KEY_QUOTE = "expect_key_q"    # 期待 "
    IN_KEY = "in_key"                    # 在写 key 字符
    EXPECT_COLON = "expect_colon"        # 期待 :
    EXPECT_VALUE_START = "expect_v_s"    # 期待 数字 或 "
    IN_VALUE_NUMBER = "in_value_number"
    IN_VALUE_STRING = "in_value_string"
    EXPECT_COMMA_OR_CLOSE = "expect_,_or_}"
    DONE = "done"


LOWER = set("abcdefghijklmnopqrstuvwxyz")
DIGIT = set("0123456789")


@dataclass
class JsonFSM:
    """跟踪当前状态, 给出"下一可接受字符集合"。"""
    state: State = State.EXPECT_OPEN
    n_pairs: int = 0                 # 已写完的 key:value 对数
    last_key_or_val_len: int = 0     # 当前 key 或 value 写了几位
    min_pairs: int = 1
    max_pairs: int = 3

    def legal_chars(self) -> Set[str]:
        if self.state == State.EXPECT_OPEN:
            return {"{"}
        if self.state == State.EXPECT_KEY_QUOTE:
            return {'"'}
        if self.state == State.IN_KEY:
            # 至少 1 字符, 达到 6 字符就强制关闭
            if self.last_key_or_val_len >= 6:
                return {'"'}
            ok = set(LOWER)
            if self.last_key_or_val_len >= 1:
                ok.add('"')
            return ok
        if self.state == State.EXPECT_COLON:
            return {":"}
        if self.state == State.EXPECT_VALUE_START:
            # JSON 不允许数字以 0 开头 (合法 0 必须独立成数, 简化掉)
            return (DIGIT - {"0"}) | {'"'}
        if self.state == State.IN_VALUE_NUMBER:
            # 数字最多 4 位, 到了就强制结束
            if self.last_key_or_val_len >= 4:
                ok = set()
                if self.n_pairs + 1 < self.max_pairs:
                    ok.add(",")
                if self.n_pairs + 1 >= self.min_pairs:
                    ok.add("}")
                return ok
            ok = set(DIGIT)
            if self.last_key_or_val_len >= 1:
                if self.n_pairs + 1 < self.max_pairs:
                    ok.add(",")
                if self.n_pairs + 1 >= self.min_pairs:
                    ok.add("}")
            return ok
        if self.state == State.IN_VALUE_STRING:
            if self.last_key_or_val_len >= 6:
                return {'"'}
            ok = set(LOWER)
            if self.last_key_or_val_len >= 1:
                ok.add('"')
            return ok
        if self.state == State.EXPECT_COMMA_OR_CLOSE:
            ok = set()
            if self.n_pairs < self.max_pairs:
                ok.add(",")
            if self.n_pairs >= self.min_pairs:
                ok.add("}")
            return ok
        if self.state == State.DONE:
            return set()
        return set()

    def advance(self, ch: str) -> None:
        s = self.state
        if s == State.EXPECT_OPEN and ch == "{":
            self.state = State.EXPECT_KEY_QUOTE
        elif s == State.EXPECT_KEY_QUOTE and ch == '"':
            self.state = State.IN_KEY
            self.last_key_or_val_len = 0
        elif s == State.IN_KEY and ch in LOWER:
            self.last_key_or_val_len += 1
        elif s == State.IN_KEY and ch == '"':
            self.state = State.EXPECT_COLON
        elif s == State.EXPECT_COLON and ch == ":":
            self.state = State.EXPECT_VALUE_START
        elif s == State.EXPECT_VALUE_START and ch in DIGIT:
            self.state = State.IN_VALUE_NUMBER
            self.last_key_or_val_len = 1
        elif s == State.EXPECT_VALUE_START and ch == '"':
            self.state = State.IN_VALUE_STRING
            self.last_key_or_val_len = 0
        elif s == State.IN_VALUE_NUMBER and ch in DIGIT:
            self.last_key_or_val_len += 1
        elif s == State.IN_VALUE_NUMBER and ch == ",":
            self.n_pairs += 1
            self.state = State.EXPECT_KEY_QUOTE
        elif s == State.IN_VALUE_NUMBER and ch == "}":
            self.n_pairs += 1
            self.state = State.DONE
        elif s == State.IN_VALUE_STRING and ch in LOWER:
            self.last_key_or_val_len += 1
        elif s == State.IN_VALUE_STRING and ch == '"':
            self.state = State.EXPECT_COMMA_OR_CLOSE
        elif s == State.EXPECT_COMMA_OR_CLOSE and ch == ",":
            self.n_pairs += 1
            self.state = State.EXPECT_KEY_QUOTE
        elif s == State.EXPECT_COMMA_OR_CLOSE and ch == "}":
            self.n_pairs += 1
            self.state = State.DONE
        else:
            raise ValueError(f"非法转移: state={s}, char={ch!r}")


def constrained_decode(
    tok: CharTokenizer, fsm: JsonFSM, max_steps: int = 50, seed: int = 0
) -> str:
    """用随机 logits 模拟模型输出, 用 fsm 强制约束。"""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(max_steps):
        legal = fsm.legal_chars()
        if not legal:
            break
        # 任意 logits (这里随机), 给所有 token 评分
        logits = rng.randn(tok.vocab_size).astype(np.float32)
        # 构造 mask: 非法字符的 logit 设 -inf
        mask = np.full(tok.vocab_size, -np.inf, dtype=np.float32)
        for ch in legal:
            if ch in tok.stoi:
                mask[tok.stoi[ch]] = 0.0
        masked = logits + mask
        # greedy 出
        tid = int(np.argmax(masked))
        ch = tok.itos[tid]
        fsm.advance(ch)
        out.append(ch)
        if fsm.state == State.DONE:
            break
    return "".join(out)


def main():
    banner("M14 - Structured Output (JSON via FSM)")

    tok = CharTokenizer()
    print(f"\nvocab size = {len(tok)}")

    # ---- 多次随机, 全部产出合法 JSON --------------------------- #
    print("\n[1] 用纯随机 logits + FSM 约束, 跑 5 次:")
    import json
    for seed in range(5):
        fsm = JsonFSM(min_pairs=1, max_pairs=3)
        s = constrained_decode(tok, fsm, max_steps=80, seed=seed)
        try:
            obj = json.loads(s)
            ok = "✓"
        except Exception as e:
            obj = e
            ok = "✗"
        print(f"  seed={seed}  {ok}  output = {s!r}")

    print("\n[2] 关掉 FSM (无约束) 看会发生什么:")
    rng = np.random.RandomState(0)
    raw = "".join(tok.itos[int(np.argmax(rng.randn(len(tok))))]
                  for _ in range(20))
    print(f"  raw output = {raw!r}")
    try:
        json.loads(raw); print("  ✓ JSON valid")
    except Exception as e:
        print(f"  ✗ JSON invalid: {e}")

    print("\n  ✓ 加 FSM 后 100% 合法; 不加几乎 100% 不合法")
    print("  这是 outlines / xgrammar 的思想原型")


if __name__ == "__main__":
    main()
