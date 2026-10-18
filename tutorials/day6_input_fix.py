"""
Day 6：修掉一个「偷偷烧钱」的 bug

day5 的 bug 在 135-136 行：
    if not user:
        continue          # ← 这句看着是"跳过"，实际是"回到循环最开头"
而循环最开头（105 行）就是「问模型」。
所以用户只按了一下回车，我们白白问了模型一次：真花钱、真等几秒、
换回一个没人要的回答。

修法：**不要在等用户的地方用 continue。**
把「等用户输入」单独抽成一个小函数，它自己转圈，
拿不到合格输入就自己重来，绝不回到主循环。

用法：
    python day6_input_fix.py
    python day6_input_fix.py "我上周买的耳机到哪了？"
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


# ==================================================================
# ★★ 本次唯一的改动：把「等用户」抽成独立函数 ★★
# ==================================================================
def wait_for_user() -> str | None:
    """
    一直等到拿到一句「能用的」用户输入。
    返回 None = 用户想退出 / 输入流断了。

    关键点：这个函数**自己带一个 while True**。
    空输入时它在这里面转圈重来，绝对不会回到主循环，
    所以一次模型都不会被调用 —— 不花钱、不等待。
    """
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if user.lower() in {"quit", "exit", "退出"}:
            return None

        if not user:
            # 空行：提醒一下，然后在这个小圈里重来
            # （这个 continue 是安全的 —— 它回到的是第 96 行，不是主循环）
            print("（空的内容没有发送，重新输入一次，或输入 quit 结束）")
            continue

        return user


first = sys.argv[1] if len(sys.argv) > 1 else "我上周买的耳机到哪了？"

messages = [
    {"role": "system", "content": "你是一名电商客服助手。"},
    {"role": "user", "content": first},
]

print(f"你：{first}")
print("（输入 quit / exit / 退出 结束）\n")

step = 1

while step <= MAX_STEPS:
    print("（客服正在输入…）", flush=True)

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
        continue  # 这里 continue 是对的：工具结果回来了，确实要再问模型

    # ---------- 情况 B：模型说话了 ----------
    print(f"\r客服：{msg.content}")
    print("-" * 46)  # 分隔线：让"你："提示落到干净的新一行

    user = wait_for_user()
    if user is None:
        print("已退出。")
        break

    messages.append({"role": "user", "content": user})

print(f"\n【本次共问了模型 {step - 1} 次】")
