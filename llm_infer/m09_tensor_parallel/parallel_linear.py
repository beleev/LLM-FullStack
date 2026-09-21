"""
parallel_linear.py — 单进程模拟 Megatron 式 Tensor Parallel 的一个 Transformer block

是什么: 把每层权重按列/行切给 tp 张卡, 每张卡只存、只算自己那 1/tp。
解决的瓶颈: 单卡显存装不下权重 (70B fp16 = 140 GB); 顺带每卡 KV cache 与算力也降到 1/tp。
关键数字: 每个 block 恰好 2 次 all-reduce (attention 后 1 次, MLP 后 1 次), 每次载荷 T×D 个激活。
切法: Q/K/V、gate/up 按列切 (输出维, 无通信) → 中间逐元素/逐头计算各卡独立 → O、down 按行切 (输入维) 再 all-reduce 求和。
    列切的边界必须落在 head 边界上: 每卡拿到 n_head/tp 个完整的头, 头内 softmax 不被切开。
读代码盯住: `tp_block` 里的两次 `comm.all_reduce`, 以及 rank 本地计算就是同一个 `mha` / `swiglu`, 只是权重更窄。
对应真实系统: vLLM / Megatron 的 ColumnParallelLinear (QKVParallelLinear, MergedColumnParallelLinear)
    + RowParallelLinear; all-reduce 由 NCCL 完成。
模拟声明: "rank" 只是 Python 列表里的一项, all-reduce 是顺序求和; 不测通信耗时, 只数次数与字节。
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

from llm_infer.core import dense_attention, rms_norm
from llm_infer.core.utils import silu

Weights = Dict[str, np.ndarray]
COL, ROW = ("q", "k", "v", "gate", "up"), ("o", "down")


def split_column(W: np.ndarray, tp: int) -> List[np.ndarray]:
    """(D_in, D_out) → tp 份 (D_in, D_out/tp)。X@W = concat_r(X@W_r), 无需通信。"""
    assert W.shape[1] % tp == 0
    return np.split(W, tp, axis=1)


def split_row(W: np.ndarray, tp: int) -> List[np.ndarray]:
    """(D_in, D_out) → tp 份 (D_in/tp, D_out)。X@W = Σ_r X_r@W_r, 求和就是 all-reduce。"""
    assert W.shape[0] % tp == 0
    return np.split(W, tp, axis=0)


class Comm:
    """模拟 NCCL all-reduce(sum), 记录次数与载荷字节。"""

    def __init__(self):
        self.n_allreduce = 0
        self.payload_bytes = 0

    def all_reduce(self, per_rank: List[np.ndarray]) -> np.ndarray:
        if len(per_rank) > 1:                             # tp=1 无需通信
            self.n_allreduce += 1
            self.payload_bytes += per_rank[0].nbytes
        return np.sum(per_rank, axis=0)


# ---- 单卡算子; TP 的 rank 本地计算复用它们, 只是传入切过的权重 ----------------

def mha(x, Wq, Wk, Wv, Wo, n_head: int) -> np.ndarray:
    """多头 causal attention。x (T,D); Wq/Wk/Wv (D, n_head·dh); Wo (n_head·dh, D) → (T,D)。"""
    T = x.shape[0]
    heads = lambda W: (x @ W).reshape(T, n_head, -1).transpose(1, 0, 2)   # (T,n_head·dh) → (n_head,T,dh)
    h = dense_attention(heads(Wq), heads(Wk), heads(Wv))                  # (n_head,T,dh), 每头独立 softmax
    h = h.astype(x.dtype)                                                 # core 基线内部会升到 fp64; 通信字节按激活 dtype 算
    return h.transpose(1, 0, 2).reshape(T, -1) @ Wo                       # (T,n_head·dh) @ (n_head·dh,D) → (T,D)


def swiglu(x, Wg, Wu, Wd) -> np.ndarray:
    """x (T,D); Wg/Wu (D,d_mlp); Wd (d_mlp,D)。silu 与 * 都逐元素, 所以 d_mlp 维可以任意切。"""
    return (silu(x @ Wg) * (x @ Wu)) @ Wd                                 # (T,d_mlp) @ (d_mlp,D) → (T,D)


def dense_block(x, W: Weights, n_head: int) -> np.ndarray:
    """不切分的基线: pre-norm 残差 block。"""
    h = x + mha(rms_norm(x, W["ln1"]), W["q"], W["k"], W["v"], W["o"], n_head)
    return h + swiglu(rms_norm(h, W["ln2"]), W["gate"], W["up"], W["down"])


# ---- Tensor Parallel ------------------------------------------------------------

def shard_weights(W: Weights, tp: int) -> List[Weights]:
    """加载期一次性切分 → 每个 rank 一份权重字典。RMSNorm 的 gamma 很小, 每卡复制一份。"""
    parts = {**{k: split_column(W[k], tp) for k in COL}, **{k: split_row(W[k], tp) for k in ROW}}
    return [{**{k: p[r] for k, p in parts.items()}, "ln1": W["ln1"], "ln2": W["ln2"]} for r in range(tp)]


def tp_block(x, ranks: List[Weights], n_head: int, comm: Comm) -> np.ndarray:
    """x (T,D) 在每个 rank 上都有完整副本 (激活不切, 只切权重)。"""
    tp = len(ranks)
    assert n_head % tp == 0, "head 是 attention 的最小切分单位"
    xn = rms_norm(x, ranks[0]["ln1"])                     # 各 rank 算出的结果相同, 模拟里只算一次
    partial = [mha(xn, R["q"], R["k"], R["v"], R["o"], n_head // tp) for R in ranks]   # 每项 (T,D): 本卡那几个头的贡献
    h = x + comm.all_reduce(partial)                      # all-reduce #1
    hn = rms_norm(h, ranks[0]["ln2"])
    partial = [swiglu(hn, R["gate"], R["up"], R["down"]) for R in ranks]              # 每项 (T,D)
    return h + comm.all_reduce(partial)                   # all-reduce #2
