"""
Day 6 主线：给 Agent 加第二个工具，兑现它「有手机号就能查」的承诺

新概念：**工具描述 = 路由规则**
两个工具摆在那里，模型靠什么决定用哪个？全靠 description 那句话。
所以两个工具的 description 要「互相指名」：
    - 有订单号时用 A，别用 B
    - 只有手机号时用 B，别硬编一个订单号

另一个改动：工具从「一个」变成「多个」后，
主循环不能再写死调用哪个函数，要改成「按名字查表分发」。

用法：
    python day6b_two_tools.py "我的手机号是13800138000，帮我查一下订单"
    python day6b_two_tools.py "我的订单 SO20240917001 到哪了？"
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

MAX_STEPS = 12

# ==================================================================
# ★★ 两个工具的「说明书」——注意它们互相指名 ★★
# ==================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "按【订单号】查询单笔订单的状态和物流信息。"
                "仅在用户提供了形如 SO20240917001 的订单号时使用。"
                "如果用户只给了手机号、没给订单号，不要调用本工具，"
                "应改用 search_orders_by_phone。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20240917001",
                        "pattern": r"^SO\d{11}$",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_orders_by_phone",
            "description": (
                "按【手机号】查询该用户名下的全部订单，返回一个列表。"
                "仅在用户提供了 11 位手机号时使用。"
                "一个手机号可能对应多笔订单，必须把全部订单都列给用户，不要只说一笔。"
                "如果用户已经给出了订单号，不要调用本工具，改用 get_order_status。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "11 位手机号，形如 13800138000",
                        "pattern": r"^1\d{10}$",
                    }
                },
                "required": ["phone"],
                "additionalProperties": False,
            },
        },
    },
]

ORDERS = {
    o["order_id"]: o
    for o in json.loads(Path("data/mock_orders.json").read_text(encoding="utf-8"))["orders"]
}


def get_order_status(order_id: str) -> str:
    """按订单号查单笔。"""
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


def search_orders_by_phone(phone: str) -> str:
    """按手机号查名下的全部订单（可能多笔）。"""
    hits = [o for o in ORDERS.values() if o.get("phone") == phone]
    if not hits:
        return json.dumps({"found": False, "phone": phone, "count": 0}, ensure_ascii=False)
    return json.dumps(
        {
            "found": True,
            "phone": phone,
            "count": len(hits),
            "orders": [
                {
                    "order_id": o["order_id"],
                    "item": o["item"],
                    "status": o["status"],
                    "amount": o.get("amount"),
                    "paid_at": o.get("paid_at"),
                }
                for o in hits
            ],
        },
        ensure_ascii=False,
    )


# ==================================================================
# ★★ 工具多了，就得「按名字查表分发」 ★★
# 以前只有 1 个工具，可以直接写 get_order_status(...)
# 现在有 2 个，不知道模型会点哪个 → 先把名字和函数做成一张表
# ==================================================================
TOOL_FUNCS = {
    "get_order_status": get_order_status,
    "search_orders_by_phone": search_orders_by_phone,
}


def run_tool(name: str, arguments: str) -> str:
    """
    执行一个工具调用，永远返回字符串（哪怕出错）。

    两个兜底很重要，面试常被问到：
      1. 模型可能调一个不存在的工具名 → 不能崩，要返回错误让它知道
      2. 模型可能吐出不合格式的 JSON → 不能崩，同上
    原则：工具出错时，把错误「告诉模型」比「让程序崩溃」好得多，
          因为模型看到错误后有机会换个方式重试。
    """
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return json.dumps(
            {"error": f"没有名为 {name} 的工具", "available": list(TOOL_FUNCS)},
            ensure_ascii=False,
        )
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"参数不是合法 JSON: {e}"}, ensure_ascii=False)

    if not isinstance(args, dict):
        return json.dumps({"error": "参数必须是一个对象"}, ensure_ascii=False)

    try:
        return fn(**args)
    except TypeError as e:
        # 模型传了不存在的参数名，或漏了必填参数
        return json.dumps({"error": f"参数不对: {e}"}, ensure_ascii=False)


def wait_for_user() -> str | None:
    while True:
        try:
            user = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if user.lower() in {"quit", "exit", "退出"}:
            return None
        if not user:
            print("（空的内容没有发送，重新输入一次，或输入 quit 结束）")
            continue
        return user


first = sys.argv[1] if len(sys.argv) > 1 else "我的手机号是13800138000，帮我查一下我的订单"

messages = [
    {"role": "system", "content": "你是一名电商客服助手。"},
    {"role": "user", "content": first},
]

print(f"你：{first}")
print("（输入 quit / exit / 退出 结束）")
print(f"【可用工具】{', '.join(TOOL_FUNCS)}\n")

step = 1

while step <= MAX_STEPS:
    print("（客服正在输入…）", flush=True)

    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    msg = resp.choices[0].message
    step += 1

    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            print(f"  [调工具] {call.function.name}({call.function.arguments})")
            print(f"  [结果]   {result[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        continue

    print(f"\r客服：{msg.content}")
    print("-" * 46)

    user = wait_for_user()
    if user is None:
        print("已退出。")
        break

    messages.append({"role": "user", "content": user})

print(f"\n【本次共问了模型 {step - 1} 次】")
