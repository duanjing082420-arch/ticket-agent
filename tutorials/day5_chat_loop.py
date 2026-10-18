"""
Day 5：把 input() 塞进循环 —— 从「寄信」变成「打电话」

和 day4_agent_loop.py 只有一处关键改动：
    day4 里「模型不要工具」时做的是 break（跑完就结束）
    这里改成：打印它的话 → input() 等用户打字 → 继续下一圈

用法：
    python day5_chat_loop.py
    python day5_chat_loop.py "我上周买的耳机到哪了？"

输入 quit / exit / 退出 结束对话。
"""

import json
import os
import sys
from pathlib import Path

from openai import OpenAI


def load_dotenv(path: str = ".env") -> None:
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

MAX_STEPS = 12  # 安全网：整个会话最多问模型 12 次

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
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

ORDERS = {
    o["order_id"]: o
    for o in json.loads(Path("data/mock_orders.json").read_text(encoding="utf-8"))["orders"]
}


def get_order_status(order_id: str) -> str:
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


# 第一句话从命令行来；后面的话都靠 input() 现场输入
first = sys.argv[1] if len(sys.argv) > 1 else "我上周买的耳机到哪了？"

messages = [
    {"role": "system", "content": "你是一名电商客服助手。"},
    {"role": "user", "content": first},
]

print(f"你：{first}")
print("（输入 quit / exit / 退出 结束）\n")

step = 1

while step <= MAX_STEPS:
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    msg = resp.choices[0].message
    step += 1  # 只要问过模型一次就计数，防止死循环

    # ---------- 情况 A：模型要调工具 ----------
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = get_order_status(args["order_id"])
            print(f"  [调工具] {call.function.name}({call.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        continue  # 回到开头，再问模型一次（这次它就拿到数据了）

    # ---------- 情况 B：模型说话了（这就是 day4 里 break 的位置）----------
    print(f"客服：{msg.content}")

    # ★ 关键改动：不 break，而是等用户打字 ★
    try:
        user = input("\n你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        break

    if user.lower() in {"quit", "exit", "退出"}:
        print("已退出。")
        break

    if not user:
        continue

    # 用户的话记进聊天记录，下一圈模型就能看见了
    messages.append({"role": "user", "content": user})

print(f"\n【本次共问了模型 {step - 1} 次】")
