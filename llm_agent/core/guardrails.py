"""护栏: 不可信数据隔离 + 密钥脱敏。

没有它: agent 读到的任何网页 / 文档都能对它"下指令" (prompt injection);
工具输出里的 API key 会原样进日志、进上下文、进下一次模型请求。
关键设计 —— 纵深防御, 不指望任何单层:
  1. 标记: 不可信工具的输出包进 <untrusted_data>, 命中注入特征再加 flag (提示模型: 这是数据不是指令)
  2. 污点: 本轮上下文一旦混入不可信数据, 高风险工具一律拒绝 (确定性, 不依赖模型听话; 在 agent.py)
  3. 脱敏: 内容进 transcript 之前先替换密钥
对应: Claude Code 把工具结果视为数据 + 对可疑结果做注入提示; "lethal trifecta" (私有数据 + 不可信内容 + 对外通道) 的切断思路。
"""

from __future__ import annotations

import re
from typing import List

# ponytail: 正则只认常见格式; 生产用 detect-secrets / gitleaks 规则集 + 熵检测
_SECRETS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[a-z0-9._-]{8,}"),
    re.compile(r"(?i)((?:password|passwd|api_key|apikey|token|secret)\s*[=:]\s*)\S+"),
]

# 特征匹配只能"标记可疑", 挡不住改写过的注入 —— 真正兜底的是第 2 层污点规则
_INJECTIONS = [
    re.compile(r"(?i)ignore (all )?(previous|prior|above) instructions"),
    re.compile(r"忽略(之前|以上|先前|上面)的?(所有)?(指令|指示)"),
    re.compile(r"(?i)you are now|new system prompt|do not tell the user"),
    re.compile(r"(?i)^\s*(agent|assistant|system)\s*[:：]", re.MULTILINE),
]


class Guardrails:
    def redact(self, text: str) -> str:
        for pattern in _SECRETS:
            # 带捕获组的规则保留 key 名 (password=), 只抹值 —— 日志仍可读
            text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
        return text

    def scan(self, text: str) -> List[str]:
        return [m.group(0).strip() for p in _INJECTIONS for m in [p.search(text)] if m]

    def wrap_untrusted(self, text: str) -> str:
        hits = self.scan(text)
        flag = f' injection_suspected="{"; ".join(hits)}"' if hits else ""
        text = text.replace("</untrusted_data", "<\\/untrusted_data")  # 文档自带闭合标签 = 想提前"越狱"出数据区
        return f"<untrusted_data{flag}>\n{text}\n</untrusted_data>"
