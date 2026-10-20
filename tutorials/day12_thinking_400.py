"""Day12：验证「思考模式默认开启」对带 tools 的多轮循环的影响。

背景（2026-10-19 核实 api-docs.deepseek.com/guides/thinking_mode）：
  1. deepseek-flash 思考模式默认开启，默认 effort = high —— 实测确认。
  2. 官方文档还写着：请求带 tools 参数时，reasoning_content 必须在后续
     每一轮完整回传，否则 API 返回 400。
     —— 实测未复现：不回传照样 200。文档这条在当前实现下没有生效，
        但别因此放松，它随时可能收紧。

本项目 agent/llm.py 与 agent/loop.py 均未处理 reasoning_content，
因此这个脚本回答两个具体问题：
  Q1 换到 deepseek-flash 之后，「调一次工具再回答」的最小循环还能不能跑通？
  Q2 思考模式开与关，输出 token 差多少？

实测结论（同一道题「9.11 和 9.8 哪个大」）：
  默认（思考开）        输出 81 token，其中 76 是思考 token
  显式关闭 thinking     输出  3 token
  —— 差 27 倍。这才是真正要命的地方：不是会不会报错，是成本。

三个对照组：
  A 默认（思考模式开，不回传 reasoning_content）  ← 项目现状
  B 显式关闭思考模式
  C 思考模式开，但按要求回传 reasoning_content

用法：
  python tutorials/day12_thinking_400.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.config import Settings, load_dotenv  # noqa: E402

load_dotenv()  # Settings.from_env() 不会自己加载 .env，必须显式调用

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "按订单号查询订单状态",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def call(api_key: str, model: str, payload: dict) -> tuple[int, dict, float]:
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return 200, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8")), time.time() - t0
        except Exception:
            return e.code, {"raw": "(非 JSON 错误体)"}, time.time() - t0


def build_assistant_message(msg: dict, keep_reasoning: bool) -> dict:
    """按是否保留 reasoning_content 组装回传给下一轮的 assistant 消息。"""
    out: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if keep_reasoning and msg.get("reasoning_content"):
        out["reasoning_content"] = msg["reasoning_content"]
    return out


def run_case(name: str, api_key: str, model: str, *, disable_thinking: bool, keep_reasoning: bool) -> None:
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")

    messages = [{"role": "user", "content": "帮我查一下订单 A1001 的状态"}]

    # —— 第一轮：模型应当发起工具调用 ——
    payload = {"model": model, "messages": messages, "tools": TOOLS}
    if disable_thinking:
        # thinking 必须放在请求体顶层。extra_body 是 OpenAI SDK 的概念，
        # SDK 发送前会把它摊平；直接用 HTTP 请求时传 extra_body 会被整体忽略
        # （实测：传了等于没传，行为与默认完全一致）。
        payload["thinking"] = {"type": "disabled"}

    code, data, dt = call(api_key, model, payload)
    print(f"  第1轮  HTTP {code}  耗时 {dt:.1f}s")
    if code != 200:
        print(f"    ✗ 失败：{json.dumps(data, ensure_ascii=False)[:300]}")
        return

    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning_content")
    tool_calls = msg.get("tool_calls")
    usage = data.get("usage", {})
    detail = usage.get("completion_tokens_details", {}) or {}

    print(f"    思考内容 reasoning_content: {'有，' + str(len(reasoning)) + ' 字' if reasoning else '无'}")
    print(f"    工具调用 tool_calls: {'有' if tool_calls else '无'}")
    print(f"    输出 token {usage.get('completion_tokens')}，其中思考 token {detail.get('reasoning_tokens', 0)}")

    if not tool_calls:
        print("    （本轮未调用工具，无法验证第二轮，跳过）")
        return

    # —— 第二轮：带上工具结果再问一次 ——
    messages.append(build_assistant_message(msg, keep_reasoning))
    messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "订单 A1001 已发货，预计明天送达"})

    payload2 = {"model": model, "messages": messages, "tools": TOOLS}
    if disable_thinking:
        payload2["thinking"] = {"type": "disabled"}

    code2, data2, dt2 = call(api_key, model, payload2)
    print(f"  第2轮  HTTP {code2}  耗时 {dt2:.1f}s")
    if code2 != 200:
        err = json.dumps(data2, ensure_ascii=False)
        print(f"    ✗ 失败：{err[:400]}")
        print("    >>> 这就是循环断掉的地方：Agent 拿不到最终答案")
        return

    final = data2["choices"][0]["message"].get("content") or ""
    print(f"    ✓ 最终回答：{final[:120]}")


def main() -> None:
    s = Settings.from_env()
    if not s.llm_api_key:
        print("缺少 LLM_API_KEY，请先配置 .env")
        raise SystemExit(1)

    print(f"模型：{s.llm_model}")
    print("场景：用户问订单状态 → 模型调工具 → 带工具结果再问一次（最小 Agent 循环）")

    run_case(
        "A 组：项目现状 —— 默认思考模式 + 不回传 reasoning_content",
        s.llm_api_key, s.llm_model,
        disable_thinking=False, keep_reasoning=False,
    )
    run_case(
        "B 组：显式关闭思考模式",
        s.llm_api_key, s.llm_model,
        disable_thinking=True, keep_reasoning=False,
    )
    run_case(
        "C 组：思考模式开，但按官方要求回传 reasoning_content",
        s.llm_api_key, s.llm_model,
        disable_thinking=False, keep_reasoning=True,
    )

    print(f"\n{'=' * 62}")
    print("怎么看结果：")
    print("  三组如果都是 200 → 循环不会断，暂时不用改代码（文档那条 400 未复现）。")
    print("  重点比「输出 token」这一行：")
    print("    A 组（默认，思考开） 的数字会明显大于 B 组（关闭思考）")
    print("    差值就是思考 token，它们按输出价计费 —— 是真实的钱。")
    print("  选 A 还是 B 取决于场景：")
    print("    工单分类、查订单、套模板回答 → B，省 90%+ 输出成本，延迟也更低")
    print("    需要多步推理的复杂决策     → A 或 reasoning_effort=low 折中")
    print("  注意：run_eval.py 的 24 条评测集如果开着思考模式跑，")
    print("        成本观测里的数字会比实际需要的虚高一个量级。")


if __name__ == "__main__":
    main()
