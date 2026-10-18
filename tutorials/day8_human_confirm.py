"""
Day 8：不可逆操作 —— 必须在代码层拦住，不能靠模型自觉

问题：如果给 Agent 一个「申请退款」的工具，它自己点了怎么办？
钱一旦退出去，收不回来。

错误做法：在提示词里写「请谨慎操作」
          → 没用。模型是非确定性的（day3 实测：同一脚本跑两次结果不一样），
             你不能指望它每次都听话。

正确做法：**代码层强制拦截**
    危险工具不直接执行，而是返回一张「待确认单」；
    主循环检测到待确认单 → 暂停 → 拿给人看 → 人点头 → 才真正执行。

顺带解决第二个问题：**幂等**
    同一笔退款被提交两次（网络重试、用户重复说），只能生效一次。
    做法：用「订单号 + 理由」算出一把钥匙（幂等键），执行过就记下来。

用法：
    python day8_human_confirm.py "帮我申请退款，订单 SO20240917001，理由是不想要了"
"""

import hashlib
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

# 记录「已经执行过」的幂等键 —— 真实项目里要存数据库，这里放内存演示
EXECUTED_KEYS: set[str] = set()

SYSTEM_PROMPT = """你是一名电商客服助手。

工作流程：
1. 先用订单号或手机号查到订单。
2. 如果用户想知道货到哪了，用订单返回的 tracking_no 继续查物流轨迹。
3. 如果用户要退款，调用 request_refund。

严格要求：
- 只报告工具真实返回的数据，不要编造。
- request_refund 需要人工确认，这不是你能决定的，照常调用即可。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "按【订单号】查询单笔订单的状态和物流信息，返回结果里含 tracking_no（运单号）。"
                "仅在用户提供了形如 SO20240917001 的订单号时使用。"
                "如果用户只给了手机号、没给订单号，改用 search_orders_by_phone。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，形如 SO20240917001"}
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
                "仅在用户提供了 11 位手机号时使用。一个手机号可能对应多笔订单。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "description": "11 位手机号"}
                },
                "required": ["phone"],
                "additionalProperties": False,
            },
        },
    },
    # ★★ 危险工具 ★★
    {
        "type": "function",
        "function": {
            "name": "request_refund",
            "description": (
                "【不可逆操作】为指定订单申请退款。"
                "仅当用户明确表达了退款意愿时使用。"
                "调用后不会立即退款，会生成一张待人工确认的单据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "要退款的订单号"},
                    "reason": {"type": "string", "description": "退款原因，用一句话概括"},
                },
                "required": ["order_id", "reason"],
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
    o = ORDERS.get(order_id)
    if not o:
        return json.dumps({"found": False, "order_id": order_id}, ensure_ascii=False)
    return json.dumps(
        {
            "found": True,
            "order_id": o["order_id"],
            "status": o["status"],
            "item": o["item"],
            "amount": o.get("amount"),
            "carrier": o["carrier"],
            "tracking_no": o["tracking_no"],
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
                {"order_id": o["order_id"], "item": o["item"], "status": o["status"]}
                for o in hits
            ],
        },
        ensure_ascii=False,
    )


def make_idempotency_key(order_id: str, reason: str) -> str:
    """
    幂等键：同一笔「订单 + 理由」算出来永远一样。
    真实项目里通常还会加上用户 ID 和时间窗口。
    """
    raw = f"refund:{order_id}:{reason.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def request_refund(order_id: str, reason: str) -> str:
    """
    ★★ 关键：这个「危险工具」不退款，只开一张待确认单 ★★
    """
    o = ORDERS.get(order_id)
    if not o:
        return json.dumps({"error": f"订单不存在：{order_id}"}, ensure_ascii=False)

    key = make_idempotency_key(order_id, reason)
    return json.dumps(
        {
            "status": "pending_confirmation",   # ← 主循环看到这个标记就会拦下来
            "action": "退款",
            "order_id": order_id,
            "item": o["item"],
            "amount": o.get("amount"),
            "reason": reason,
            "idempotency_key": key,
        },
        ensure_ascii=False,
    )


def execute_refund(payload: dict) -> str:
    """真正执行退款 —— 只有人工点了确认才会走到这里。"""
    key = payload["idempotency_key"]

    if key in EXECUTED_KEYS:
        # ★ 幂等：同样的请求第二次进来，直接跳过
        return json.dumps(
            {"status": "skipped", "detail": "这笔退款已经处理过，本次未重复执行"},
            ensure_ascii=False,
        )

    EXECUTED_KEYS.add(key)
    # 真实项目里这里会调支付网关、写数据库、发通知
    return json.dumps(
        {
            "status": "done",
            "detail": f"已为订单 {payload['order_id']} 退款 {payload['amount']} 元",
        },
        ensure_ascii=False,
    )


TOOL_FUNCS = {
    "get_order_status": get_order_status,
    "search_orders_by_phone": search_orders_by_phone,
    "request_refund": request_refund,
}


def run_tool(name: str, arguments: str) -> str:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return json.dumps({"error": f"没有名为 {name} 的工具"}, ensure_ascii=False)
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"参数不是合法 JSON: {e}"}, ensure_ascii=False)
    try:
        return fn(**args)
    except TypeError as e:
        return json.dumps({"error": f"参数不对: {e}"}, ensure_ascii=False)


def ask_human(payload: dict) -> str:
    """
    ★★ 拦截点：把待确认单拿给人看，等人点头 ★★
    真实产品里这里不是 input()，而是把单据推进审核队列，
    由后台人工点击「通过 / 驳回」。
    """
    print("\n" + "=" * 52)
    print("⚠  拦截到一个不可逆操作，需要人工确认")
    print("=" * 52)
    print(f"  动作    ：{payload['action']}")
    print(f"  订单号  ：{payload['order_id']}")
    print(f"  商品    ：{payload['item']}")
    print(f"  金额    ：{payload['amount']} 元")
    print(f"  原因    ：{payload['reason']}")
    print(f"  幂等键  ：{payload['idempotency_key']}")
    print("=" * 52)

    while True:
        ans = input("  确认执行？输入 y 执行 / n 驳回：").strip().lower()
        if ans in {"y", "yes", "是"}:
            print(f"  → 人工已确认，执行中…")
            return execute_refund(payload)
        if ans in {"n", "no", "否"}:
            print("  → 人工已驳回")
            return json.dumps({"status": "cancelled", "detail": "人工驳回，未退款"}, ensure_ascii=False)
        print("  （只认 y 或 n）")


def wait_for_user() -> str | None:
    while True:
        try:
            user = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if user.lower() in {"quit", "exit", "退出"}:
            return None
        if not user:
            print("（空的内容没有发送，重新输入一次）")
            continue
        return user


# 直接验证幂等：绕开模型，连着执行两次完全一样的退款请求
if "--demo-idempotent" in sys.argv:
    payload = {
        "action": "退款",
        "order_id": "SO20240917001",
        "item": "无线降噪耳机 Pro",
        "amount": 899.0,
        "reason": "不想要了",
        "idempotency_key": make_idempotency_key("SO20240917001", "不想要了"),
    }
    print(f"幂等键：{payload['idempotency_key']}\n")
    print("第 1 次执行 →", execute_refund(payload))
    print("第 2 次执行 →", execute_refund(payload))
    print("第 3 次执行 →", execute_refund(payload))
    print("\n看明白了吗：三次请求，只退了一次款。")
    print("幂等靠的是代码里的 EXECUTED_KEYS，不是靠模型记得住。")
    raise SystemExit(0)

first = sys.argv[1] if len(sys.argv) > 1 else "帮我申请退款，订单 SO20240917001，理由是不想要了"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": first},
]

print(f"你：{first}")
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

            # ★★★ 拦截逻辑：工具返回了「待确认」标记就拦下来 ★★★
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict) and parsed.get("status") == "pending_confirmation":
                result = ask_human(parsed)

            print(f"  [结果]   {result[:160]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        continue

    print(f"\r客服：{msg.content}")
    print("-" * 52)

    user = wait_for_user()
    if user is None:
        print("已退出。")
        break

    messages.append({"role": "user", "content": user})

print(f"\n【本次共问了模型 {step - 1} 次】")
print(f"【已执行的退款幂等键】{sorted(EXECUTED_KEYS) or '（无）'}")
