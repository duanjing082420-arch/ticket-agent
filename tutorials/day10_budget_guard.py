"""
Day 10 主线：给循环加三道预算锁

    MAX_STEPS   ← 已经有了（最多跑几圈）
    token 预算   ← 新增
    墙钟时间      ← 新增
    金额上限      ← 新增（单价从环境变量读，不填就只统计 token）

Part 1 不联网、不花钱（模拟）
Part 2 真实跑一次，打印预算账本
"""

import os
import sys
import time
from pathlib import Path

from openai import OpenAI


# 本脚本已被归档到 tutorials/，但 .env 与 data/ 都在项目根目录，
# 所以路径统一基于「本文件的上两级」推导，从任意目录启动都能找到。
ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path=None) -> None:
    """沿用 day3 的土办法读 .env：一行一行拆 key=value（不额外装依赖）"""
    p = Path(path or (ROOT / ".env"))
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_dotenv()


# ============================================================
# Part 1：为什么 MAX_STEPS 不够 —— 成本是「二次」的
# ============================================================
# 关键事实：每一圈都要把「到目前的全部 messages」重新发一遍。
# 所以第 N 圈的输入 = 前面所有圈的内容之和 → 越往后越贵。

SYSTEM_TOKENS = 30      # system prompt（每圈都要重发）
QUESTION_TOKENS = 20    # 用户的问题
ASK_TOKENS = 25         # 模型每圈的"思考+措辞"输出
TOOL_RESULT_TOKENS = 150  # 工具返回的结果（很肥）


def simulate(rounds: int):
    print(f"\n  假设每一圈都调一次工具，工具结果 {TOOL_RESULT_TOKENS} token：\n")
    print(f"  {'圈':<4}{'本次输入':>10}{'本次输出':>10}{'累计总额':>12}")
    print("  " + "-" * 40)

    history = SYSTEM_TOKENS + QUESTION_TOKENS   # 第 1 圈开始前已有的内容
    total_in = 0
    total_out = 0

    for i in range(1, rounds + 1):
        this_in = history                        # ← 全部历史重新发一遍
        this_out = ASK_TOKENS
        total_in += this_in
        total_out += this_out
        # 这一圈产出的东西，下一圈要继续背著
        history += this_out + TOOL_RESULT_TOKENS
        print(f"  {i:<4}{this_in:>10}{this_out:>10}{total_in + total_out:>12}")

    return total_in + total_out


def part1():
    print("=" * 66)
    print("  Part 1：成本是怎么涨的（纯模拟，不花钱）")
    print("=" * 66)

    t3 = simulate(3)
    t5 = simulate(5)

    print(f"\n  跑到第 3 圈的累计：{t3} token")
    print(f"  跑到第 5 圈的累计：{t5} token")
    print(f"\n  圈数只多了 2 圈（+67%），成本多了 {t5 - t3} token（+{(t5-t3)/t3*100:.0f}%）")
    print("""
  ↑ 这就是「二次增长」：圈数是线性加的，成本是越滚越快的。
    所以光靠 MAX_STEPS 卡圈数，卡不住钱 —— 必须再卡 token 总数。""")


# ============================================================
# Part 2：真实跑一次，带三道锁
# ============================================================
# 预算配置
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "6000"))     # token 上限
TIME_BUDGET = float(os.getenv("TIME_BUDGET", "45"))        # 秒
MONEY_BUDGET = float(os.getenv("MONEY_BUDGET", "0"))       # 元，0 = 不启用
# 单价不内置（各家随时调价，以官方定价页为准）；不填就只统计 token
PRICE_IN = float(os.getenv("PRICE_PER_1K_INPUT", "0"))     # 元 / 千 token 输入
PRICE_OUT = float(os.getenv("PRICE_PER_1K_OUTPUT", "0"))   # 元 / 千 token 输出


class Budget:
    """三道锁的账本"""

    def __init__(self):
        self.total_in = 0
        self.total_out = 0
        self.start = time.time()
        self.stopped_by = None

    def cost(self) -> float:
        return self.total_in / 1000 * PRICE_IN + self.total_out / 1000 * PRICE_OUT

    def elapsed(self) -> float:
        return time.time() - self.start

    # ★ 关键：检查发生在「发请求之前」，不是发完之后
    def check_before_call(self, estimated_input: int):
        if self.total_in + estimated_input > TOKEN_BUDGET:
            self.stopped_by = f"token 预算耗尽（已用 {self.total_in}，再加这次 {estimated_input} > {TOKEN_BUDGET}）"
            return False
        if self.elapsed() > TIME_BUDGET:
            self.stopped_by = f"超时（已用 {self.elapsed():.1f}s > {TIME_BUDGET}s）"
            return False
        if MONEY_BUDGET > 0 and self.cost() > MONEY_BUDGET:
            self.stopped_by = f"金额超上限（已花 {self.cost():.4f} 元 > {MONEY_BUDGET} 元）"
            return False
        return True

    def add(self, usage):
        if usage:
            self.total_in += getattr(usage, "prompt_tokens", 0) or 0
            self.total_out += getattr(usage, "completion_tokens", 0) or 0

    def summary(self):
        print(f"\n  ── 预算账本 ──")
        print(f"  累计输入 {self.total_in} token ／ 累计输出 {self.total_out} token")
        print(f"  合计     {self.total_in + self.total_out} token（预算 {TOKEN_BUDGET}）")
        print(f"  耗时     {self.elapsed():.2f}s（上限 {TIME_BUDGET}s）")
        if PRICE_IN or PRICE_OUT:
            print(f"  花费     {self.cost():.4f} 元")
        else:
            print(f"  花费     （未配置单价，只统计 token）")
        if self.stopped_by:
            print(f"  ⚠ 触发了预算锁：{self.stopped_by}")


# ---------- 工具（沿用 day7 那三个，简化版只留一个做演示）----------
import json

ORDERS = json.loads((ROOT / "data" / "mock_orders.json").read_text(encoding="utf-8"))["orders"]
TRACKING = json.loads((ROOT / "data" / "mock_tracking.json").read_text(encoding="utf-8"))


def get_order_status(order_id: str) -> str:
    for o in ORDERS:
        if o["order_id"] == order_id:
            return json.dumps({k: o.get(k) for k in
                               ("order_id", "status", "item", "carrier", "tracking_no", "shipped_at")},
                              ensure_ascii=False)
    return json.dumps({"found": False}, ensure_ascii=False)


def get_tracking_detail(tracking_no: str) -> str:
    t = TRACKING.get(tracking_no)
    if not t:
        return json.dumps({"found": False}, ensure_ascii=False)
    return json.dumps({"found": True, **t}, ensure_ascii=False)


TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "按订单号查询订单状态和运单号。仅在用户提供了订单号时使用。",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "订单号，形如 SO20240917001"}},
            "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_tracking_detail",
        "description": "按运单号查询物流轨迹。拿到 tracking_no 之后再调用。",
        "parameters": {"type": "object", "properties": {
            "tracking_no": {"type": "string", "description": "运单号"}},
            "required": ["tracking_no"], "additionalProperties": False}}},
]
FUNCS = {"get_order_status": get_order_status, "get_tracking_detail": get_tracking_detail}


def part2(question: str):
    print("\n" + "=" * 66)
    print("  Part 2：真实跑一次（带三道锁）")
    print("=" * 66)

    client = OpenAI(api_key=os.getenv("LLM_API_KEY"),
                    base_url=os.getenv("LLM_BASE_URL"),
                    timeout=30.0)

    messages = [
        {"role": "system", "content": "你是客服助手。查到 tracking_no 后继续调用 get_tracking_detail。"
                                      "只报告工具真实返回的数据，不要编造。"},
        {"role": "user", "content": question},
    ]
    budget = Budget()

    for step in range(1, MAX_STEPS + 1):
        # ★ 发请求之前先估算这次要发多少（粗算：字符数）
        est = sum(len(str(m.get("content") or "")) for m in messages)
        if not budget.check_before_call(est):
            print(f"\n  [第 {step} 圈] 被预算锁拦下，不再请求模型")
            break

        print(f"\n  [第 {step} 圈] 请求模型…（预计输入约 {est} token）")
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            messages=messages, tools=TOOLS, temperature=0.2,
        )
        budget.add(resp.usage)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"  客服：{msg.content}")
            break

        for tc in msg.tool_calls:
            print(f"    [调工具] {tc.function.name}({tc.function.arguments})")
            result = FUNCS[tc.function.name](**json.loads(tc.function.arguments))
            print(f"    [结果]   {result[:90]}…")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    budget.summary()


if __name__ == "__main__":
    part1()
    q = sys.argv[1] if len(sys.argv) > 1 else "我的订单 SO20240917001 到哪了？"
    part2(q)
