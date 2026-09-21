"""最小 MCP server: stdio 上的 JSON-RPC 2.0, 每行一条消息。

独立脚本 —— 不 import 本包任何东西 (真实的 MCP server 可以是任何语言写的另一个进程)。
只实现 tools 能力的三个方法: initialize / tools/list / tools/call。
注意: stdout 是协议通道, 日志只能写 stderr, 否则客户端会把日志当成坏掉的 JSON。
"""

import json
import sys

TOOLS = [
    {
        "name": "get_weather",
        "description": "Return a (fake) weather report for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
]


def call_tool(name, args):
    if name == "get_weather":
        return f"{args['city']}: sunny, 24C, light wind"
    if name == "add":
        return str(args["a"] + args["b"])
    raise KeyError(name)


def handle(req):
    method, params = req.get("method"), req.get("params") or {}
    if method == "initialize":
        return {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "toy-weather", "version": "0.1"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        try:
            text, is_error = call_tool(params["name"], params.get("arguments") or {}), False
        except Exception as exc:  # 工具失败是"正常结果" (isError), 不是协议错误
            text, is_error = f"tool error: {exc!r}", True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}
    raise LookupError(method)


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if "id" not in req:  # notification (如 notifications/initialized): 规范要求不回复
            continue
        try:
            resp = {"jsonrpc": "2.0", "id": req["id"], "result": handle(req)}
        except LookupError:
            resp = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32601, "message": "Method not found"}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()  # 不 flush, 客户端会永远等下去


if __name__ == "__main__":
    main()
