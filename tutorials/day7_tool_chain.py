"""
Day 7：让 Agent 一次对话里连调两个工具 —— 工具链

回顾问题：「查到订单后，自动再查一次物流，需要改什么？」

答案是两样，而且**循环代码一行都不用动**（day4 那个 continue 早就支持了）：
    1. 加一个新工具：get_tracking_detail（按运单号查轨迹）
    2. 改 system prompt：告诉它「查到运单号后，继续查」

这就是本文件的核心概念 —— **工具链**：
    第 1 次工具调用的结果里，藏着第 2 次调用需要的参数。
    模型必须先拿到订单 → 才知道运单号 → 才能查物流。
这种「后一步依赖前一步」的链式调用，才是 Agent 真正区别于单次调用的地方。

用法：
    python day7_tool_chain.py "我的订单 SO20240917001 到哪了？"
    python day7_tool_chain.py "我手机号13800138000，买的耳机到哪了"
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
# ★★ 改动 2：system prompt 里写清「工作流程」 ★★
# 光有工具不够，你得告诉它：查完订单之后还要再查一步
# ==================================================================
SYSTEM_PROMPT = """你是一名电商客服助手。

工作流程：
1. 先用订单号或手机号查到订单。
2. 如果用户想知道货到哪了，就用订单里返回的 tracking_no（运单号）
   继续调用 get_tracking_detail 查询详细物流轨迹。
3. 订单状态是「待发货」时没有运单号，不要调用物流查询，直接告诉用户还没发货。

严格要求：
只报告工具真实返回的数据，不要编造任何物流信息。
不要向用户承诺工具做不到的事。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "按【订单号】查询单笔订单的状态和物流信息，返回结果里含 tracking_no（运单号）。"
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
    # ★★ 改动 1：第三个工具，按运单号查轨迹 ★★
    {
        "type": "function",
        "function": {
            "name": "get_tracking_detail",
            "description": (
                "按【运单号】查询详细的物流轨迹（走过哪些站点、当前状态、预计送达时间）。"
                "必须先通过 get_order_status 或 search_orders_by_phone 拿到 tracking_no 后再调用。"
                "订单状态为「待发货」时没有运单号，不要调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_no": {
                        "type": "string",
                        "description": "快递运单号，形如 SF1234567890",
                    }
                },
                "required": ["tracking_no"],
                "additionalProperties": False,
            },
        },
    },
]

ORDERS = {
    o["order_id"]: o
    for o in json.loads(Path("data/mock_orders.json").read_text(encoding="utf-8"))["orders"]
}
TRACKINGS = json.loads(Path("data/mock_tracking.json").read_text(encoding="utf-8"))


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


def search_orders_by_phone(phone: str) -> str:
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


def get_tracking_detail(tracking_no: str) -> str:
    t = TRACKINGS.get(tracking_no)
    if not t:
        return json.dumps({"found": False, "tracking_no": tracking_no}, ensure_ascii=False)
    return json.dumps(
        {
            "found": True,
            "carrier": t["carrier"],
            "tracking_no": t["tracking_no"],
            "current_status": t["current_status"],
            "eta": t["eta"],
            "events": t["events"],
        },
        ensure_ascii=False,
    )


TOOL_FUNCS = {
    "get_order_status": get_order_status,
    "search_orders_by_phone": search_orders_by_phone,
    "get_tracking_detail": get_tracking_detail,
}


def run_tool(name: str, arguments: str) -> str:
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


first = sys.argv[1] if len(sys.argv) > 1 else "我的订单 SO20240917001 到哪了？"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": first},
]

print(f"你：{first}")
print("（输入 quit / exit / 退出 结束）")
print(f"【可用工具】{', '.join(TOOL_FUNCS)}\n")

step = 1

while step <= MAX_STEPS:
    print(f"（第 {step} 次问模型…）", flush=True)

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
            print(f"  [结果]   {result[:160]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        continue  # ← 循环代码一行没改，它本来就支持连调

    print(f"\r客服：{msg.content}")
    print("-" * 46)

    user = wait_for_user()
    if user is None:
        print("已退出。")
        break

    messages.append({"role": "user", "content": user})

print(f"\n【本次共问了模型 {step - 1} 次】")
