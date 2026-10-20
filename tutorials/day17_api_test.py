"""day17：验证 FastAPI 服务化的核心设计 —— L2 人工确认拆成两段式 + 幂等。

先确保服务已启动（另一个终端）：
    uvicorn server:app --reload --port 8000

跑本脚本：
    python tutorials/day17_api_test.py

要看的三件事：
1. POST /chat 遇到 L2 危险动作（建工单）时，返回 behavior=needs_confirm 和 pending，
   **但不真正创建工单** —— 「AI 拟稿，不代燃」。
2. POST /confirm 把 pending 原样回传后，才真正执行、生成工单号。
3. 同一条 pending 再 confirm 一次，返回同一个结果（幂等键生效），不会重复建单。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def show_chat(tag: str, d: dict) -> None:
    print(f"  behavior      : {d.get('behavior')}")
    print(f"  intent        : {d.get('intent')}")
    print(f"  tools_called  : {d.get('tools_called')}")
    print(f"  steps         : {d.get('steps')}  |  stop_reason: {d.get('stop_reason')}")
    print(f"  latency_ms    : {d.get('latency_ms')}  |  cost: {d.get('cost')}")
    out = (d.get("output") or "").replace("\n", " ")
    print(f"  output        : {out[:150]}")
    pendings = d.get("pending") or []
    print(f"  pending 数量  : {len(pendings)}")
    for p in pendings:
        print(f"      ├─ preview        : {str(p.get('preview'))[:80]}")
        print(f"      └─ idempotency_key: {p.get('idempotency_key')}")
    print()


def main() -> int:
    print("=" * 66)
    print("FastAPI 服务化验证：L2 人工确认两段式 + 幂等")
    print("=" * 66)

    try:
        health = get("/health")
    except urllib.error.URLError as exc:
        print(f"❌ 连不上服务 {BASE}：{exc}")
        print("   请先在另一个终端启动：uvicorn server:app --reload --port 8000")
        return 1
    print(f"[0] GET /health -> {health}\n")

    uid = "api-day17-user"
    action_text = "我要退款，订单号 SO20240905004，蓝牙音箱连不上手机"

    # ---- 第 1 步：问一个会触发 L2 建工单的请求 ----
    print(f"[1] POST /chat  {action_text!r}")
    print("    期望：behavior=needs_confirm，pending 非空，但工单还没真创建")
    r1 = post("/chat", {"text": action_text, "user_id": uid})
    show_chat("first", r1)

    pendings = r1.get("pending") or []
    if not pendings:
        print("⚠️  本次没有返回 pending —— 模型可能选择了别的处理路径（如追问/直接回答）。")
        print("   这本身也是合法行为；重跑一次通常就能拿到 pending。可以就此结束。")
        return 0

    pending = pendings[0]

    # ---- 第 2 步：模拟用户点了「确认」 ----
    print("[2] POST /confirm  （模拟用户在前端点了「确认执行」）")
    print("    期望：真正在执行，返回工单号")
    c1 = post("/confirm", {"user_id": uid, "pending": pending})
    print(f"    ok     : {c1.get('ok')}")
    print(f"    result : {json.dumps(c1.get('result'), ensure_ascii=False)[:260]}\n")

    # ---- 第 3 步：同一条 pending 再提交一次（幂等检验）----
    print("[3] POST /confirm  （同一条 pending 重复提交 —— 幂等检验）")
    print("    期望：返回与上面相同的结果/工单号，不会新建一张工单")
    c2 = post("/confirm", {"user_id": uid, "pending": pending})
    print(f"    ok     : {c2.get('ok')}")
    print(f"    result : {json.dumps(c2.get('result'), ensure_ascii=False)[:260]}\n")

    # 判定标准要抓「本质」而不是整个返回包：
    # 第一次 executed=true 并新建工单；第二次 executed=false（幂等键命中）但**返回同一个工单号**。
    # 所以要比的是 ticket_id 是否相同，而不是整个 dict 是否相等
    # （直接比整个 dict 会把 executed 标志的差异误判成失败 —— 断言写得太粗的典型例子）。
    def ticket_id_of(c: dict) -> str:
        return str(((c.get("result") or {}).get("result") or {}).get("ticket_id", ""))

    t1, t2 = ticket_id_of(c1), ticket_id_of(c2)
    executed_first = bool((c1.get("result") or {}).get("executed"))
    executed_again = bool((c2.get("result") or {}).get("executed"))

    print("=" * 66)
    print(f"第一次返回工单号 : {t1}  | executed={executed_first}")
    print(f"第二次返回工单号 : {t2}  | executed={executed_again}")
    # 幂等的本质 = 「不会产生第二张工单」，所以真正的判据是两次工单号相同。
    # executed=false 表示「这一次没有真的再执行」，正是保护生效的证据，不是失败。
    # 注：幂等键缓存在运行中的服务进程里（窗口 300s）。若刚跑过一次又立刻重跑，
    # 第一次 confirm 也可能命中旧键而 executed=false —— 这同样是正确行为。
    if t1 and t1 == t2:
        print("✅ 幂等验证通过：重复提交未产生第二张工单，两次返回同一张。")
        if not executed_again:
            print("   且第二次 executed=false —— 幂等键拦截生效，未重复执行。")
    else:
        print("⚠️  两次工单号不同 —— 可能重复建单，请核对该幂等实现。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
