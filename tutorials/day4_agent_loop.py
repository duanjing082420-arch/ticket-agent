"""
Day 4：把「一次性脚本」改成「循环」—— 这才是真正的 Agent

和 day3_first_tool.py 的区别：
  day3 是手写了两遍「问模型」，跑完就结束。
  这里改成循环：问模型 → 它要工具就执行 → 回去再问 → 直到它不要工具为止。

用法：
    python day4_agent_loop.py
    python day4_agent_loop.py "我上周买的耳机到哪了？"
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

# ==================================================================
# 安全网：最多跑几圈
#   为什么必须有：模型可能永远「再查一个」，那就是死循环。
#   死循环 = 每次都要花钱 + 用户一直等。这是面试必考点「有界自主」。
# ==================================================================
MAX_STEPS = 5

# ============ 工具说明书（和 Day 3 一模一样，没变）============
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

# ============ 工具真身（和 Day 3 一模一样，没变）============
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


question = sys.argv[1] if len(sys.argv) > 1 else "我的订单 SO20240917001 现在到哪了？"

# messages 是「我们和模型的全部聊天记录」，循环里会不断往里加东西，
# 每一圈都把整包重新发过去。
messages = [
    {"role": "system", "content": "你是一名电商客服助手。"},
    {"role": "user", "content": question},
]

print(f"【问题】{question}")
print(f"【规则】最多跑 {MAX_STEPS} 圈；模型不再要工具就停\n")

final_answer = None  # 用来装最终答案
step = 1

# ==================================================================
# 循环开始 —— 注意下面这段整体缩进 4 格，Python 靠缩进判断「哪些行在圈里」
# ==================================================================
while step <= MAX_STEPS:
    print(f"---------- 第 {step} 圈 ----------")

    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
    )
    msg = resp.choices[0].message

    # ★ 停止条件就在这里 ★
    if not msg.tool_calls:
        print("模型没再要工具 → 它觉得可以回答了 → 跳出循环")
        final_answer = msg.content
        break  # break = 跳出这个圈，继续往下走（不是结束程序）

    # 模型要了工具：先把它那句「我要调工具」原样记进聊天记录
    messages.append(msg)

    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_order_status(args["order_id"])
        print(f"  调工具 → {call.function.name}({call.function.arguments})")
        print(f"  查到   → {result}")

        # 把查到的结果记进聊天记录，下一圈模型就能看见了
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    step += 1  # 跑完一圈，计数器 +1
    # 到这里缩进还没结束，所以自动回到 while 那一行，再来一圈

# ==================================================================
# 循环结束（缩进回到最左边了）
# ==================================================================
if final_answer is None:
    print(f"\n[超过 {MAX_STEPS} 圈还没办完，被安全网强行停下 —— 这种情况要转人工]")
else:
    print(f"\n【最终回答】\n{final_answer}")

print(f"\n【一共跑了 {step} 圈】")
