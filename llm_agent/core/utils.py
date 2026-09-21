"""demo 共用的小工具: 打印、分词、token 估算。"""

from __future__ import annotations

import re
from typing import List


def banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def kv(key: str, value: object) -> None:
    print(f"  {key:<24}: {value}")


def shorten(text: str, width: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


_CJK = r"一-鿿"


def tokenize(text: str) -> List[str]:
    """英文按词, 中文按字符 bigram。

    中文没有空格, 只认 [a-z0-9]+ 会把中文整段丢掉 (查询得分恒为 0);
    bigram 是不依赖分词器的经典折中 (Lucene CJKAnalyzer 同款)。
    """
    tokens = []
    for run in re.findall(rf"[a-z0-9]+|[{_CJK}]+", text.lower()):
        if re.match(rf"[{_CJK}]", run):
            tokens.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        else:
            tokens.append(run)
    return tokens


def estimate_tokens(text: str) -> int:
    # ponytail: 粗估 (英文 ~4 字符/token, 中文 ~1 字/token); 真实系统用 messages.count_tokens
    cjk = len(re.findall(rf"[{_CJK}]", text))
    return cjk + (len(text) - cjk + 3) // 4
