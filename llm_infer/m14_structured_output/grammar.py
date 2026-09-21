"""
grammar.py — 结构化输出: 字符级 FSM → 预编译成 token 级 mask 表 (xgrammar / outlines 的核心思想)

是什么: 采样前把"会让输出违反语法"的 token 的 logit 置 -inf, 模型只能说合法的话。
瓶颈  : 真实词表是多字符 token (BPE), 每步对 V 个 token 逐字符试走 FSM 是 O(V·len) 的
        纯 CPU 开销, 卡在 GPU 前向和采样之间 → 直接吃掉 decode 延迟。
解法  : FSM 状态有限 → 离线把每个状态下"哪些 token 整段字符都能走通、走完落在哪个状态"
        预编译成表, 在线每步只做一次查表 O(1)。
盯住  : `next_state[s, t]` (S,V) — 状态 s 吃下 token t 后的状态, -1 = 走不通;
        `mask_table = next_state >= 0`; `need[s, t]` — 选了 t 之后最少还要几个 token 才能收尾。
对应  : outlines 的 index (state → allowed token ids); xgrammar 的 adaptive token mask cache
        (它处理 CFG/下推自动机, 只能预编译"与栈无关"的 token, 其余运行时再查);
        vLLM 的 guided decoding / structured outputs 后端。

迷你 grammar (JSON 子集, 无嵌套):
    object = "{" pair ("," pair)* "}"        pair 数 ∈ [min_pairs, max_pairs]
    pair   = string ":" value                ":" 与 "," 之后允许一个可选空格
    value  = string | number | "true" | "false" | "null"
    string = '"' [a-z]{1,6} '"'              number = [1-9][0-9]{0,3}
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np

from llm_infer.core import TinyLM, softmax


class State(Enum):
    EXPECT_OPEN = "{"
    EXPECT_KEY_QUOTE = 'key"'
    IN_KEY = "key"
    EXPECT_COLON = ":"
    EXPECT_VALUE_START = "value?"
    IN_VALUE_NUMBER = "number"
    IN_VALUE_STRING = "string"
    IN_LITERAL = "literal"               # true / false / null 写到一半
    EXPECT_COMMA_OR_CLOSE = ",}"
    DONE = "done"


LOWER = set("abcdefghijklmnopqrstuvwxyz")
DIGIT = set("0123456789")
LITERALS = {"t": "rue", "f": "alse", "n": "ull"}     # 首字符 → 剩余字符
_COUNTING = (State.IN_KEY, State.IN_VALUE_NUMBER, State.IN_VALUE_STRING)


@dataclass
class JsonFSM:
    """字符级状态机: legal_chars() 给出下一步可接受字符, advance(ch) 前进一步。

    带计数器 (n_pairs / length), 但计数有上界 → 全部配置 key() 仍是有限集, 可以枚举成 DFA。
    """
    state: State = State.EXPECT_OPEN
    n_pairs: int = 0          # 已被 "," 或 "}" 结算的 pair 数 (正在写的这一对还没算进去)
    length: int = 0           # 当前 key / value 已写字符数
    lit: str = ""             # 字面量还没写完的部分, 如 "ue"
    spaced: bool = False      # 可选空格已用掉
    min_pairs: int = 1
    max_pairs: int = 3

    def key(self) -> Tuple:
        if self.state == State.DONE:                     # 写了几对都是同一个接受态
            return (State.DONE,)
        return (self.state, self.n_pairs, self.length, self.lit, self.spaced)

    def _separators(self) -> Set[str]:
        """value 写完后能接什么。所有 value 类型共用这一处 (若 string 分支用 n_pairs、
        number 分支用 n_pairs+1, 导致 string 结尾时被迫 ≥2 对且可能超过 max_pairs)。"""
        n = self.n_pairs + 1                         # 算上刚写完、尚未结算的这一对
        return ({","} if n < self.max_pairs else set()) | ({"}"} if n >= self.min_pairs else set())

    def legal_chars(self) -> Set[str]:
        s, space = self.state, (set() if self.spaced else {" "})
        if s == State.EXPECT_OPEN:
            return {"{"}
        if s == State.EXPECT_KEY_QUOTE:
            return {'"'} | space
        if s in (State.IN_KEY, State.IN_VALUE_STRING):   # 1~6 个小写字母
            return (LOWER if self.length < 6 else set()) | ({'"'} if self.length >= 1 else set())
        if s == State.EXPECT_COLON:
            return {":"}
        if s == State.EXPECT_VALUE_START:                # 不允许前导 0 (JSON 规定)
            return (DIGIT - {"0"}) | {'"'} | set(LITERALS) | space
        if s == State.IN_VALUE_NUMBER:                   # 最多 4 位; 数字没有闭合符, 直接接分隔符
            return (DIGIT if self.length < 4 else set()) | self._separators()
        if s == State.IN_LITERAL:
            return {self.lit[0]}
        if s == State.EXPECT_COMMA_OR_CLOSE:
            return self._separators()
        return set()                                     # DONE

    def advance(self, ch: str) -> None:
        if ch not in self.legal_chars():
            raise ValueError(f"非法转移: state={self.state}, char={ch!r}")
        s = self.state
        if ch == " ":
            self.spaced = True
            return
        self.spaced = False
        if s == State.EXPECT_OPEN:
            self.state = State.EXPECT_KEY_QUOTE
        elif s == State.EXPECT_KEY_QUOTE:
            self.state = State.IN_KEY
        elif s == State.EXPECT_COLON:
            self.state = State.EXPECT_VALUE_START
        elif s == State.EXPECT_VALUE_START:
            if ch == '"':
                self.state = State.IN_VALUE_STRING
            elif ch in DIGIT:
                self.state, self.length = State.IN_VALUE_NUMBER, 1
            else:
                self.state, self.lit = State.IN_LITERAL, LITERALS[ch]
        elif s == State.IN_LITERAL:
            self.lit = self.lit[1:]
            if not self.lit:
                self.state = State.EXPECT_COMMA_OR_CLOSE
        elif ch in (",", "}"):                           # 来自 IN_VALUE_NUMBER 或 EXPECT_COMMA_OR_CLOSE
            self.n_pairs += 1
            self.state = State.EXPECT_KEY_QUOTE if ch == "," else State.DONE
        elif ch == '"':                                  # 闭合 key / string value
            self.state = State.EXPECT_COLON if s == State.IN_KEY else State.EXPECT_COMMA_OR_CLOSE
        else:                                            # key / string / number 里再写一个字符
            self.length += 1
        if self.state not in _COUNTING:
            self.length = 0                              # 计数器归零 → 等价配置合并, DFA 状态更少


# ------------------------------------------------------------------ #
# 1) 字符级 FSM → 显式 DFA                                             #
# ------------------------------------------------------------------ #

def compile_char_dfa(fsm0: JsonFSM) -> Tuple[List[Dict[str, int]], int]:
    """BFS 枚举所有可达配置 → trans[s] = {ch: s'}; 返回 (trans, 接受态 id)。初始态 id = 0。"""
    ids = {fsm0.key(): 0}
    frontier, trans, accept = [fsm0], [], -1
    while frontier:
        nxt = []
        for f in frontier:                               # frontier 顺序 == id 顺序, trans 可直接 append
            row = {}
            for ch in sorted(f.legal_chars()):
                g = copy.copy(f)
                g.advance(ch)
                if g.key() not in ids:
                    ids[g.key()] = len(ids)
                    nxt.append(g)
                row[ch] = ids[g.key()]
            trans.append(row)
            if f.state == State.DONE:
                accept = ids[f.key()]
        frontier = nxt
    return trans, accept


# ------------------------------------------------------------------ #
# 2) 多字符词表 + token 级预编译                                        #
# ------------------------------------------------------------------ #

EOS = "<eos>"


def build_vocab() -> List[str]:
    """≤128 个字符串 token: 单字符 + 类 BPE 的多字符片段 (含跨语法边界的, 如 '":' / 'e"')。"""
    single = list('{}":, ') + sorted(LOWER) + sorted(DIGIT)
    multi = ['{"', '":', '": ', '",', '", "', '"}', '":"', '","', ', "', '""',
             "true", "false", "null", "tru", "nul",
             "10", "12", "20", "25", "42", "99", "100", "00", "07", "1,", "1}", "5}",
             "na", "me", "id", "ag", "er", "in", "on", "th", "age", "key", "val", "name", "value",
             'e"', 'd"', 'e":', 'id":']
    junk = ["[", "]", "\n", "}}", "{{", "::", ",,", "'", "\\"]   # 在本 grammar 下永远非法
    vocab = [EOS] + single + multi + junk
    assert len(vocab) == len(set(vocab)) <= 128
    return vocab


def token_row(trans: List[Dict[str, int]], accept: int, vocab: List[str], s: int) -> np.ndarray:
    """状态 s 下逐 token 逐字符试走 → (V,) int, 走通则是落点状态, 否则 -1。

    这就是"在线现算"的 O(V·len) 做法; 预编译只是把它对每个状态各跑一遍存起来。
    """
    row = np.full(len(vocab), -1, dtype=np.int32)
    for t, piece in enumerate(vocab):
        if piece == EOS:                                 # EOS 只在接受态合法 → 输出一定是完整 JSON
            row[t] = accept if s == accept else -1
            continue
        cur = s
        for ch in piece:
            cur = trans[cur].get(ch, -1)
            if cur < 0:
                break
        row[t] = cur
    return row


@dataclass
class TokenTable:
    vocab: List[str]
    next_state: np.ndarray    # (S, V) int32, -1 = 非法
    mask_table: np.ndarray    # (S, V) bool
    need: np.ndarray          # (S, V) 选 t 之后最少还需几个 token 才能以 EOS 收尾; 非法 = inf
    accept: int
    eos_id: int


def compile_token_table(fsm0: JsonFSM, vocab: List[str]) -> TokenTable:
    trans, accept = compile_char_dfa(fsm0)
    next_state = np.stack([token_row(trans, accept, vocab, s) for s in range(len(trans))])  # (S, V)
    mask_table = next_state >= 0
    eos_id = vocab.index(EOS)

    # dist[s] = 从 s 出发最少几个 token 能结束 (含 EOS); token 图上的最短路, 迭代到不动点
    dist = np.full(len(trans), np.inf)
    dist[accept] = 1
    while True:
        via = np.where(mask_table, dist[next_state], np.inf)     # (S, V) 走 t 之后还要多少
        new = np.minimum(dist, via.min(axis=1) + 1)
        if np.array_equal(new, dist):
            break
        dist = new
    need = np.where(mask_table, dist[next_state], np.inf)        # (S, V)
    need[accept, eos_id] = 0                                     # EOS 之后什么都不需要
    return TokenTable(vocab, next_state, mask_table, need, accept, eos_id)


# ------------------------------------------------------------------ #
# 3) 约束解码: logits → mask → 采样 → FSM 前进                          #
# ------------------------------------------------------------------ #

def generate(lm: TinyLM, table: TokenTable, max_tokens: int, rng: np.random.RandomState,
             mode: str = "budget", temperature: float = 1.0) -> Tuple[str, bool]:
    """返回 (文本, 是否以 EOS 正常结束)。mode:
        "none"   不加约束
        "mask"   只查 mask_table — 语法合法, 但 max_tokens 截断时 JSON 不完整 (vLLM 等的默认行为)
        "budget" 再用 need 表筛掉"选了就来不及收尾"的 token — 预算内必定闭合
    temperature=0 → greedy。
    """
    s, kv, tok, out = 0, None, table.eos_id, []          # 借 EOS 当 BOS
    assert mode != "budget" or table.need[0].min() < max_tokens, "max_tokens 连最短合法输出都放不下"
    for step in range(max_tokens):
        logits, kv = lm.forward([tok], kv)
        logits = logits[0].astype(np.float64)            # (V,)
        if mode != "none":
            ok = table.mask_table[s]                     # (V,) bool — O(1) 查表, 与 token 长度无关
            if mode == "budget":
                ok = ok & (table.need[s] <= max_tokens - step - 1)   # 本步之后还剩几个 token
            logits = np.where(ok, logits, -np.inf)
        if temperature == 0:
            tok = int(np.argmax(logits))
        else:
            tok = int(rng.choice(len(logits), p=softmax(logits / temperature)))
        if tok == table.eos_id:
            return "".join(out), True
        out.append(table.vocab[tok])
        if mode != "none":
            s = int(table.next_state[s, tok])
    return "".join(out), False
