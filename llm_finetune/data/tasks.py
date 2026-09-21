"""
可学习、依赖 prompt 的合成任务 — 全章所有方法共用的 "数据集 + 判分器"

是什么: prompt = L 个随机 token, 正确回复 = f(prompt) (copy / reverse / sort) 再接 EOS。
解决什么: 随机 token + 随机偏好只能检验 "背得下一个 batch"; 这里答案由 prompt 决定,
          训练集 / 留出集按 prompt 的 token 和 mod 5 严格不相交 → 可以报告**留出集**指标。
序列布局:  [x_1 … x_L  SEP | y_1 … y_L  EOS]     prompt 段长 P = L+1 (含 SEP), 回复段长 R = L+1 (含 EOS)
三种用法:  SFT → 留出集 exact-match;  偏好 → 正确回复 vs 损坏回复;  RLVR → `verify()` 程序化判分。
读代码时盯住: `make_labels` —— 全章唯一做 "右移一位 + prompt mask" 的地方。
"""

from typing import Tuple

import torch

PAD, SEP, EOS = 0, 1, 2
FIRST_SYMBOL = 3          # 内容 token 取 [3, V)
IGNORE = -100


def make_labels(seq: torch.Tensor, prompt_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seq [B, S] (右 pad) → idx [B, S], labels [B, S], 只监督回复段。idx 就是整条 seq (含 EOS: reward model 要读到它)。

    labels 已经左移一位: labels[:, t] = seq[:, t+1], 即 "位置 t 的输出要预测的 token"; 最后一列没有下一个 token, 填 -100。
        位置 t     :   0    …  P-1 |  P   …  S-2   S-1
        idx[t]     :  x_1   …  SEP | y_1  …  y_L   EOS
        labels[t]  :  -100  …  y_1 | y_2  …  EOS   -100     ← labels[:, P-1] = y_1 是**第一个回复 token**
    所以 prompt mask 是 labels[:, :P-1], 不是 [:, :P] —— 后者把 y_1 也盖掉:
    模型永远学不到 "看完 prompt 该说的第一个词", DPO 也丢掉了 log p(y_1|x) 这一项
    (上下文相同不代表这一项相同: 目标 token y_1 在 chosen / rejected 里可以不同)。
    """
    labels = torch.cat([seq[:, 1:], torch.full_like(seq[:, :1], IGNORE)], dim=1)   # [B, S] 左移一位, 末列无目标
    labels[:, : prompt_len - 1] = IGNORE                # 只盖 prompt 内部的转移
    labels[labels == PAD] = IGNORE                      # 右 pad 不计 loss
    idx = seq
    # 被监督的位置数必须恰好等于回复长度 (含 EOS); 差一位的 bug 会在这里立刻炸
    n_response = (seq[:, prompt_len:] != PAD).sum(dim=1)
    assert torch.equal((labels != IGNORE).sum(dim=1), n_response), "prompt mask 差一位"
    return idx, labels


class SeqTask:
    """
    Args:
        kind:       "copy" | "reverse" | "sort"
        vocab_size: V; 内容 token 有 V-3 种
        length:     L; prompt 空间大小 (V-3)^L, 默认 13^6 ≈ 4.8M, 训练不可能靠背
    """

    def __init__(self, kind: str = "reverse", vocab_size: int = 16, length: int = 6) -> None:
        if kind not in ("copy", "reverse", "sort"):
            raise ValueError(f"未知任务 {kind}")
        self.kind, self.vocab_size, self.length = kind, vocab_size, length
        self.prompt_len = length + 1        # P, 含 SEP
        self.response_len = length + 1      # R, 含 EOS

    # ---------------- 采样 ----------------
    def sample_prompts(self, n: int, split: str = "train") -> torch.Tensor:
        """[n, P]。留出集 = token 和 ≡ 0 (mod 5) 的 prompt, 训练集 = 其余 → 两者不相交。"""
        keep = []
        while sum(len(k) for k in keep) < n:
            x = torch.randint(FIRST_SYMBOL, self.vocab_size, (4 * n, self.length))
            is_test = x.sum(dim=1) % 5 == 0
            keep.append(x[is_test if split == "test" else ~is_test])
        x = torch.cat(keep)[:n]
        return torch.cat([x, torch.full((n, 1), SEP)], dim=1)

    def target(self, prompts: torch.Tensor) -> torch.Tensor:
        """prompts [n, P] → 正确回复 [n, R] (含 EOS)。"""
        x = prompts[:, : self.length]
        y = {"copy": x, "reverse": x.flip(1), "sort": x.sort(dim=1).values}[self.kind]
        return torch.cat([y, torch.full((len(x), 1), EOS)], dim=1)

    def corrupt(self, response: torch.Tensor) -> torch.Tensor:
        """
        造 rejected 回复 [n, R]: 一半样本 "改错一个 token", 另一半 "漏掉一个 token" (变短, 末尾补 PAD)。
        变短的那一半让 batch 里出现右 pad —— reward model 取 "最后一个非 pad 位置" 就靠它来测。
        """
        n, R = response.shape
        bad = response.clone()
        pos = torch.randint(0, R - 1, (n,))                       # 不动 EOS
        rows = torch.arange(n)
        # 换成另一个内容 token: 在 V-3 个符号上加一个非零偏移再取模, 保证一定不等于原 token
        n_sym = self.vocab_size - FIRST_SYMBOL
        shift = torch.randint(1, n_sym, (n,))
        bad[rows, pos] = (bad[rows, pos] - FIRST_SYMBOL + shift) % n_sym + FIRST_SYMBOL
        drop = torch.rand(n) < 0.5
        for i in torch.nonzero(drop).flatten().tolist():           # 删掉 pos 处 token, 左移, 末尾补 PAD
            p = int(pos[i])
            bad[i] = torch.cat([response[i, :p], response[i, p + 1:], torch.tensor([PAD])])
        return bad

    # ---------------- 判分 (RLVR 的 verifier) ----------------
    def verify(self, prompts: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
        """
        completions [n, C≥R] → reward [n] ∈ {0, 1}: 前 R 个 token 与正确回复完全一致 (含 EOS) 才得 1 分。
        纯规则、依赖 prompt: 策略不看 prompt 就拿不到分, 也没有 reward model 可以被 hack。
        """
        return (completions[:, : self.response_len] == self.target(prompts)).all(dim=1).float()

    @torch.no_grad()
    def exact_match(self, model, n: int = 256, split: str = "test", seed: int = 1234,
                    temperature: float = 0.0) -> float:
        """
        留出集 exact-match; 固定 seed → 每次评的是同一批 prompt。
        temperature=0: 贪心 (模型 "最有把握的答案" 对不对);  =1: 按策略采样 (RL 真正在优化的量, 即 pass@1)。
        """
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        prompts = self.sample_prompts(n, split)
        out = model.generate(prompts, max_new_tokens=self.response_len, temperature=temperature)
        torch.set_rng_state(state)                                # 评估不扰动训练的随机流
        return float(self.verify(prompts, out[:, self.prompt_len:]).mean())


def completion_mask(completions: torch.Tensor) -> torch.Tensor:
    """[n, C] → bool [n, C]: 第一个 EOS (含) 之前为 True。EOS 之后的 token 不是 "回复" 的一部分。"""
    after_eos = (completions == EOS).long().cumsum(dim=1) - (completions == EOS).long()
    return after_eos == 0
