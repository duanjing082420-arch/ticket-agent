"""day19：验证 LangGraph 版自主循环与手写循环行为一致。

为什么必须做这个对照：
把控制流从 for 循环换成状态图，最大的风险不是"跑不起来"，而是"跑起来了但行为悄悄变了"——
比如某道刹车晚触发一步、pending 少收集一个。这类回归在功能测试里极难发现，
但会直接改变 Agent 的停止原因与是否转人工。

做法：同一批输入，分别用两种实现各跑一遍，逐字段比对
steps / tools_called / stop_reason / pending 条数。

跑法：
    .venv/Scripts/python.exe tutorials/day19_langgraph_parity.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Settings, load_dotenv  # noqa: E402
from agent.graph_loop import run_graph_loop  # noqa: E402
from agent.llm import LLMClient  # noqa: E402
from agent.loop import run_loop  # noqa: E402
from agent.observability import setup_logging  # noqa: E402
from agent.prompts import SYSTEM_PROMPT  # noqa: E402

# 覆盖四类典型走向，其中第 2 条专门用来触发 L2 人工确认刹车
CASES = [
    ("查订单", "我手机号 13800138000，帮我查下订单"),
    ("L2 建工单(应触发人工确认)", "我要退款，订单号 SO20240905004，蓝牙音箱连不上手机"),
    ("政策咨询(应触发 RAG 检索)", "鞋子不合适能退吗"),
    ("查物流", "订单 SO20240905004 的物流到哪了"),
]

COMPARE_FIELDS = ["steps", "tools_called", "stop_reason", "pending_count"]


def snapshot(result) -> dict:
    return {
        "steps": result.steps,
        "tools_called": sorted(result.tools_called),
        "stop_reason": result.stop_reason,
        "pending_count": len(result.pending),
    }


def main() -> int:
    load_dotenv()
    setup_logging("ERROR")
    # 温度归零：两遍调用都用确定性设置，避免把模型随机性误判成实现差异
    settings = dataclasses.replace(Settings.from_env(), llm_temperature=0.0)
    llm = LLMClient(settings)

    print("=" * 72)
    print("LangGraph 版循环 vs 手写循环 —— 行为等价性对照")
    print("=" * 72)

    mismatches = 0
    for label, text in CASES:
        print(f"\n【{label}】{text}")
        msgs_plain = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        msgs_graph = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

        r_plain = run_loop(llm, msgs_plain, {"user_id": "parity-plain"}, settings)
        r_graph = run_graph_loop(llm, msgs_graph, {"user_id": "parity-graph"}, settings)

        a, b = snapshot(r_plain), snapshot(r_graph)
        ok = all(a[f] == b[f] for f in COMPARE_FIELDS)
        mismatches += 0 if ok else 1

        for f in COMPARE_FIELDS:
            flag = "  " if a[f] == b[f] else "❌"
            print(f"   {flag} {f:<14} 手写={a[f]!r}")
            if a[f] != b[f]:
                print(f"       {'':<14} 图版={b[f]!r}")
        print(f"   -> {'一致 ✅' if ok else '不一致 ❌'}")

    print("\n" + "=" * 72)
    if mismatches == 0:
        print(f"✅ {len(CASES)} / {len(CASES)} 条走向完全一致：图实现未引入行为回归。")
    else:
        print(f"❌ {mismatches} / {len(CASES)} 条不一致 —— 禁止切换，先修等价性。")
    print("=" * 72)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
