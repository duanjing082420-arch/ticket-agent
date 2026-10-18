"""
Day 3：给 AI 装第一个工具

核心看一件事：AI 到底有没有自己去查数据？
 —— 没有。它只说「我想调用」，真正动手查的是我们的代码。

用法：
    python day3_first_tool.py
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI


def load_dotenv(path: str = ".env") -> None:
    """最土的办法读 .env：一行一行拆 key=value，塞进环境变量。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    timeout=30.0,
)
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ==================================================================
# 上半边：工具的「说明书」
#   —— 这就是你说的「告诉 AI 可以调用这个文件」
#   —— 它只是一段 JSON 文字，不是代码。AI 读了它才知道有这个东西。
# ==================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            # description 决定「AI 什么时候会想到用它」，这是最关键的一句。
            "description": "按订单号查询订单的当前状态和物流信息。当用户提供了订单号并想知道订单情况时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20240917001",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
]

# ==================================================================
# 下半边：工具的「真身」
#   —— 这就是你说的「一段能查看文件内容的代码」
#   —— 真正动手查数据的，是这一段。AI 碰不到它。
# ==================================================================
ORDERS = {
    o["order_id"]: o
    for o in json.loads(Path("data/mock_orders.json").read_text(encoding="utf-8"))["orders"]
}


def get_order_status(order_id: str) -> str:
    """真实执行查询。注意：这个函数从头到尾都没被 AI 碰到过。"""
    o = ORDERS.get(order_id)
    if not o:
        return json.dumps({"found": False, "order_id": order_id}, ensure_ascii=False)
    return json.dumps(
        {
            "found": True,
            "order_id": o["order_id"],
            "status": o["status"],
            "item": o["item"],
            "carrier": o["carrier"],
            "tracking_no": o["tracking_no"],
            "shipped_at": o.get("shipped_at"),
        },
        ensure_ascii=False,
    )


# 允许从命令行换问题，方便反复试验不同问法：
#   python day3_first_tool.py "我上周买的耳机到哪了？"
# 不传就用下面这个默认问题。
question = sys.argv[1] if len(sys.argv) > 1 else "我的订单 SO20240917001 现在到哪了？"
print(f"\n【你的问题】{question}\n")

messages = [
    {"role": "system", "content": "你是一名电商客服助手。"},
    {"role": "user", "content": question},
]

# ============ 第 1 次请求：把问题 + 工具说明书 一起发过去 ============
print("=" * 60)
print("第 1 步：把「问题」和「工具说明书」一起发给模型")
print("=" * 60)

resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",  # 让模型自己决定用不用工具
)
msg = resp.choices[0].message

print("\n--- 模型这次有没有直接回答？看它说的话 ---")
print(f"content（说给人听的话） = {msg.content!r}")
print(f"tool_calls（想调用的工具） = {msg.tool_calls!r}")

if not msg.tool_calls:
    print("\n[模型没调用工具，直接回答了。把上面 content 的内容发给我]")
    raise SystemExit(0)

call = msg.tool_calls[0]
print("\n★★★ 关键观察点 ★★★")
print("模型没有去查任何东西，它只是回了这么一句：")
print(f'    「请帮我调用 {call.function.name}，参数是 {call.function.arguments}」')

# ============ 我们的代码真的去查 ============
print("\n" + "=" * 60)
print("第 2 步：我们的代码拿着参数，真的去查文件")
print("=" * 60)

args = json.loads(call.function.arguments)
result = get_order_status(args["order_id"])
print(f"\n查到的结果：{result}")

# ============ 第 3 步：把结果塞回给模型，让它组织成人话 ============
print("\n" + "=" * 60)
print("第 3 步：把查到的结果还给模型，让它说人话")
print("=" * 60)

messages.append(msg)  # 模型那句「我想调用工具」也要留着，否则它对不上号
messages.append(
    {"role": "tool", "tool_call_id": call.id, "content": result}
)

resp2 = client.chat.completions.create(model=MODEL, messages=messages)
final = resp2.choices[0].message.content

print(f"\n【模型最终回答】\n{final}")
print("\n" + "=" * 60)
print("结论：数据是我们的 Python 查的，模型只负责「决定查什么」和「把结果说成人话」。")
print("=" * 60)
