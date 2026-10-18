"""
Day 11：可观测性 —— trace + span + 脱敏

要解决的问题：
  现在排查问题只能靠屏幕上那几行 print。
  线上要能回答：「某个用户昨天下午 3 点那次失败请求，卡在第几步？」

三个新东西：
  1. trace_id  一次请求的身份证，贯穿这一次请求里的所有步骤
  2. span      里面的一小段（问一次模型、执行一个工具都算一个 span）
  3. 脱敏      写进日志之前，手机号这类敏感信息必须打码

用法：
  python day11_trace.py                    完整演示（正常 + 故障 + 事后还原）
  python day11_trace.py "你的问题"           只跑一条
"""

import json
import os
import re
import sys
import time
import uuid
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
MAX_STEPS = 6

# ==================================================================
# 下面这些和 day7 完全一样：system prompt、工具定义、三个查询函数
# 今天讲的不是它们，所以原样复用，不多解释
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
                "只给手机号时不要调用，改用 search_orders_by_phone。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20240917001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_orders_by_phone",
            "description": (
                "按【手机号】查询该名下的所有订单，返回一个列表（可能有多笔）。"
                "仅在用户提供了 11 位手机号、且没有给订单号时使用。"
                "如果用户给了订单号，优先用 get_order_status。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "11 位手机号，形如 13800138000",
                    }
                },
                "required": ["phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tracking_detail",
            "description": (
                "按【运单号】查询详细物流轨迹（走过哪些站点、当前状态、预计送达时间）。"
                "必须先通过前两个工具拿到 tracking_no 才能调用。"
                "订单未发货时没有运单号，不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_no": {
                        "type": "string",
                        "description": "运单号，形如 SF1234567890",
                    }
                },
                "required": ["tracking_no"],
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


# ==================================================================
# ★★ 今天的新东西 1：脱敏 ★★
#
# 日志会落到磁盘、会被同步到日志平台、会被很多人看到。
# 手机号这种东西一旦进去就删不干净，所以必须在「写出去之前」打码。
# ==================================================================
_PHONE_RE = re.compile(r"(1[3-9]\d)\d{4}(\d{4})")


def mask(text: str) -> str:
    """把文本里的手机号中段打码：13800138000 → 138****8000"""
    return _PHONE_RE.sub(r"\1****\2", text)


# ==================================================================
# ★★ 今天的新东西 2：Trace / Span ★★
#
# trace = 一次请求的完整记录（一个 trace_id）
# span  = 里面的一小步（问一次模型、执行一个工具，各是一个 span）
# ==================================================================
class Trace:
    def __init__(self, question: str):
        self.trace_id = "cs-" + uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.question = question
        self.spans: list[dict] = []
        self.logs: list[dict] = []

    # ---- span 开始：记下名字和起始时间，返回一个编号 ----
    def start(self, name: str, **meta) -> int:
        self.spans.append(
            {"name": name, "t0": time.time(), "t1": None, "status": "running", **meta}
        )
        return len(self.spans) - 1

    # ---- span 结束：补上结束时间、耗时、成功与否 ----
    def end(self, idx: int, status: str = "ok", **meta):
        s = self.spans[idx]
        s["t1"] = time.time()
        s["duration_ms"] = int((s["t1"] - s["t0"]) * 1000)
        s["status"] = status
        s.update(meta)

    # ---- 写一条日志：所有内容过一遍脱敏 ----
    def emit(self, level: str, event: str, **fields):
        rec = {
            "ts": time.strftime("%H:%M:%S"),
            "level": level,
            "trace_id": self.trace_id,
            "event": event,
        }
        for k, v in fields.items():
            rec[k] = mask(v) if isinstance(v, str) else v
        self.logs.append(rec)
        print("[LOG] " + json.dumps(rec, ensure_ascii=False))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "question": mask(self.question),
            "spans": [
                {k: v for k, v in s.items() if k not in ("t0", "t1")} for s in self.spans
            ],
            "logs": self.logs,
        }


def run_agent(question: str, fail_tool: str | None = None) -> tuple[Trace, str]:
    """跑一次 Agent。fail_tool 用来人为制造故障，演示日志能不能定位到那一步。"""
    tr = Trace(question)
    tr.emit("INFO", "request_start", question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    final = ""

    for step in range(1, MAX_STEPS + 1):
        # ---- span 包住「问模型」这一步 ----
        i = tr.start("llm_call", step=step)
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, temperature=0.2, timeout=30.0
            )
        except Exception as e:
            tr.end(i, status="error", error=type(e).__name__)
            tr.emit("ERROR", "llm_call_failed", step=str(step), error=str(e)[:120])
            final = f"（调用失败：{type(e).__name__}）"
            break

        u = resp.usage
        tr.end(
            i,
            status="ok",
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            final = msg.content or ""
            tr.emit("INFO", "final_answer", preview=(msg.content or "")[:60])
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            # ---- span 包住「执行工具」这一步 ----
            j = tr.start(f"tool:{tc.function.name}", step=step)
            args = tc.function.arguments
            if fail_tool and tc.function.name == fail_tool:
                result = json.dumps(
                    {"error": "模拟故障：下游物流接口返回 500"}, ensure_ascii=False
                )
                tr.end(j, status="error", error="downstream_500")
            else:
                result = run_tool(tc.function.name, args)
                tr.end(j, status="ok", result_len=len(result))
            tr.emit("INFO", "tool_result", tool=tc.function.name, args=args, result=result[:90])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    tr.emit("INFO", "request_end", total_spans=str(len(tr.spans)))
    return tr, final


def show_timeline(tr: Trace):
    """事后还原：只有 trace_id，把这次请求完整还原出来"""
    print(f"\n  ── trace {tr.trace_id} 的时间线 ──")
    print(f"  {'#':<3}{'耗时':>9}  {'状态':<7}步骤")
    print("  " + "-" * 42)
    for n, s in enumerate(tr.spans, 1):
        flag = "  " if s["status"] == "ok" else "← "  # 失败的标出来
        print(f"  {n:<3}{s.get('duration_ms', 0):>7}ms  {s['status']:<7}{flag}{s['name']}")
    if tr.spans:
        slowest = max(tr.spans, key=lambda s: s.get("duration_ms", 0))
        print(f"\n  最慢的一步：{slowest['name']}（{slowest.get('duration_ms', 0)}ms）")
    bad = [s for s in tr.spans if s["status"] != "ok"]
    if bad:
        print(f"  失败的步骤：{[s['name'] for s in bad]}")


# ==================================================================
def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        tr, ans = run_agent(sys.argv[1])
        print(f"\n【回答】{ans}")
        show_timeline(tr)
        return

    q = "我手机号13800138000，买的耳机到哪了"

    print("=" * 66)
    print("  Part 1：正常跑一次，看它吐出来的日志长什么样")
    print("=" * 66)
    tr1, ans1 = run_agent(q)
    print(f"\n【回答】{ans1[:150]}")
    show_timeline(tr1)

    print("\n" + "=" * 66)
    print("  Part 2：故意让「查物流」这一步挂掉，还能不能定位")
    print("=" * 66)
    tr2, ans2 = run_agent(q, fail_tool="get_tracking_detail")
    print(f"\n【回答】{ans2[:150]}")
    show_timeline(tr2)

    # ---- 日志落盘：真实线上就是这么干的（一行一条 JSON） ----
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    with (log_dir / "traces.jsonl").open("a", encoding="utf-8") as f:
        for tr in (tr1, tr2):
            f.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")

    print("\n" + "=" * 66)
    print("  Part 3：事后还原 —— 只给你一个 trace_id，能查出什么")
    print("=" * 66)
    print(f"\n  上面两条已写入 logs/traces.jsonl（共 {2} 条）")
    print(f"  拿着 {tr2.trace_id} 去查：\n")
    for line in (log_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["trace_id"] == tr2.trace_id:
            print(f"  时间        {rec['started_at']}")
            print(f"  用户问的是  {rec['question']}")
            print(f"  一共走了    {len(rec['spans'])} 步")
            for s in rec["spans"]:
                if s["status"] != "ok":
                    print(f"  ⚠ 卡在这一步  {s['name']}（step {s.get('step')}）→ {s.get('error')}")
            print(f"  日志条数    {len(rec['logs'])}")

    print("\n" + "=" * 66)
    print("  Part 4：脱敏 —— 不做的后果")
    print("=" * 66)
    raw = '{"phone": "13800138000", "count": 2}'
    print(f"\n  直接写进日志：{raw}")
    print(f"  过一遍脱敏：  {mask(raw)}")
    print("\n  日志会落盘、会被同步、会被很多人看到。")
    print("  手机号一旦进去就删不干净，所以要在「写出去之前」打码，不是事后补救。")


if __name__ == "__main__":
    main()
