"""
lora.py — Multi-LoRA serving: 一份底模权重, 同一个 batch 里每条请求用各自的 LoRA adapter。

瓶颈: 每个客户一份微调模型 → 显存放不下 N 份底模; 把 LoRA 合并进 W 又使不同 adapter 的请求无法同 batch (吞吐)。
做法: 不合并。y = x·Wᵀ (全 batch 共享, 1 次 gemm) + scale·(x·Aᵀ)·Bᵀ (按每个 token 的 adapter id 取各自的 A/B)。
命名 (LoRA 论文 / PEFT 约定): A 是**降维** (r, d_in), 随机初始化; B 是**升维** (d_out, r), **零初始化**; ΔW = B·A (d_out, d_in)。
关键数字: adapter 参数量 r·(d_in+d_out) vs 底模 d_in·d_out; d=4096, r=16 → 0.78%。
读代码盯住: `tok_adapter` (N_tok,) — 每个 token 属于哪个 adapter; BGMV 用它 gather, SGMV 用它分段。
真实系统: Punica (BGMV/SGMV kernel), S-LoRA (adapter 分页显存), vLLM `--enable-lora` 的 punica wrapper, SGLang LoRA backend。

W 在本模块按 nn.Linear 的布局存成 (d_out, d_in), 行向量 x 的前向是 x @ W.T — 这样 ΔW = B @ A 与论文公式逐字对应。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class LoRAAdapter:
    A: np.ndarray               # (r, d_in)   down-projection, 随机初始化
    B: np.ndarray               # (d_out, r)  up-projection,   零初始化 → 训练开始时 ΔW = 0, 模型行为与底模完全一致
    alpha: float = 16.0

    @property
    def r(self) -> int:
        return self.A.shape[0]

    @property
    def scale(self) -> float:
        return self.alpha / self.r

    def delta_w(self) -> np.ndarray:
        return self.scale * self.B @ self.A                 # (d_out, r) @ (r, d_in) → (d_out, d_in)


def init_adapter(d_in: int, d_out: int, r: int, seed: int) -> LoRAAdapter:
    rs = np.random.RandomState(seed)
    return LoRAAdapter(A=(rs.randn(r, d_in) / np.sqrt(d_in)).astype(np.float32),
                       B=np.zeros((d_out, r), np.float32))


def merged_forward(xs: List[np.ndarray], ids: List[int], W: np.ndarray, adapters: List[LoRAAdapter]):
    """参考答案: 每条请求用合并后的权重 W' = W + scale·B·A。数学定义本身, 但每个 adapter 要一份 (d_out, d_in) 的 W'。"""
    return [x @ (W + adapters[a].delta_w()).T for x, a in zip(xs, ids)]      # (T_i, d_in) @ (d_in, d_out) → (T_i, d_out)


def loop_forward(xs: List[np.ndarray], ids: List[int], W: np.ndarray, adapters: List[LoRAAdapter]):
    """不合并, 但逐请求算: n_req 次底模 gemm, 每次 batch 很小 → GPU 利用率低。"""
    outs = []
    for x, a in zip(xs, ids):
        ad = adapters[a]
        outs.append(x @ W.T + ad.scale * (x @ ad.A.T) @ ad.B.T)             # (T_i, r) 是中间的低秩瓶颈
    return outs


def _flatten(xs: List[np.ndarray], ids: List[int]):
    x = np.concatenate(xs, axis=0)                                           # (N_tok, d_in) 所有请求的 token 拼成一个大 batch
    tok_adapter = np.repeat(ids, [len(v) for v in xs])                       # (N_tok,) 每个 token 的 adapter id
    return x, tok_adapter, np.cumsum([len(v) for v in xs])[:-1]


def bgmv_forward(xs: List[np.ndarray], ids: List[int], W: np.ndarray, adapters: List[LoRAAdapter]):
    """BGMV (batched gather mat-vec): 每个 token 按 adapter id gather 出自己的 A/B, 做一次批量 mat-vec。

    要求所有 adapter 同 rank (才能堆成一个张量)。numpy 的 fancy-index 会真的拷贝出 (N_tok, r, d_in);
    Punica 的 CUDA kernel 不拷贝, 线程直接用 tok_adapter[n] 去 A_all 里寻址。
    """
    x, tok_adapter, splits = _flatten(xs, ids)
    A_all = np.stack([ad.A for ad in adapters])                              # (n_adapter, r, d_in)
    B_all = np.stack([ad.B * ad.scale for ad in adapters])                   # (n_adapter, d_out, r), scale 预乘进 B
    base = x @ W.T                                                           # (N_tok, d_out) 底模只做 1 次 gemm
    h = np.einsum("nri,ni->nr", A_all[tok_adapter], x)                       # shrink: (N_tok, r, d_in)·(N_tok, d_in) → (N_tok, r)
    delta = np.einsum("nor,nr->no", B_all[tok_adapter], h)                   # expand: (N_tok, d_out, r)·(N_tok, r) → (N_tok, d_out)
    return np.split(base + delta, splits)


def sgmv_forward(xs: List[np.ndarray], ids: List[int], W: np.ndarray, adapters: List[LoRAAdapter]):
    """SGMV (segmented gather mat-vec, Punica 论文的叫法; 每段实际是一次 gemm): 把同 adapter 的 token 归成一段, 每段一次 gemm。prefill (每请求很多 token) 时比 BGMV 划算。

    这里用 Python 循环遍历段; 真实 kernel 把所有段放进一次 grouped-GEMM launch。
    """
    x, tok_adapter, splits = _flatten(xs, ids)
    out = x @ W.T                                                            # (N_tok, d_out)
    for a in np.unique(tok_adapter):
        seg = tok_adapter == a                                               # (N_tok,) bool, 该 adapter 的全部 token (可来自多条请求)
        ad = adapters[a]
        out[seg] += ad.scale * (x[seg] @ ad.A.T) @ ad.B.T                    # (n_seg, d_in) → (n_seg, r) → (n_seg, d_out)
    return np.split(out, splits)
