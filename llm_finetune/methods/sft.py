"""
SFT — 全参监督微调 (InstructGPT 三阶段的第一步)

是什么: 在 (prompt, response) 上继续做 next-token 预测, 但 loss 只算在 response 上。
解决什么: 预训练模型只会 "续写"; SFT 教它 "看到问题就回答" —— 问题是人写的, 不该成为生成目标。
核心公式:  L = − Σ_{t ∈ response} log p(y_t | x, y_<t)          (prompt / pad 位置 label = −100 被跳过)
对齐图 (labels 已左移一位, 第 t 列 = "位置 t 的输出要预测谁"):
        idx    :  x1    x2    SEP   y1    y2    EOS
        labels : −100  −100   y1    y2    EOS   −100     ← SEP 这一列**必须**监督: 它预测的是第一个回复 token
读代码时盯住: 这里没有新 loss。SFT 与预训练的差异 100% 在 labels 里 (data/tasks.py::make_labels)。
"""

from llm_models.training.loss import StandardLMLoss

# SFT 的 loss 就是带 ignore_index=-100 的交叉熵, 与预训练逐行相同 —— 所以直接用同一个类, 不再复制一份。
SFTLoss = StandardLMLoss
