"""LangGraph 版自主循环：把 loop.py 的四道刹车画成图上的条件边。

为什么再写一遍（不是为了用框架而用框架）：
- loop.py 的刹车散落在 for 循环体的各个 if/break 里，要回答"什么时候会停"
  得通读整个函数；画成图后，四道刹车变成四条可见的边，位置一目了然。
- 代价是引入一个较重的依赖，且图的执行细节不如 for 循环直观。
  所以本文件**不替换** loop.py：两者返回完全相同的 LoopResult，
  由 service.py 按 AGENT_LOOP_IMPL 环境变量切换，可随时回滚；
  langgraph 导入失败时自动降级回手写循环。

四道刹车在图上的落点：
    1. 墙钟上限 + 步数上限 -> guard 节点（每轮 think 之前检查）
    2. token 预算           -> think 节点之后（LLM 一返回就结算，越界立刻停）
    3. 无进展               -> act 节点之后（重复动作累计到 2 次）
    4. 人工确认             -> act 节点之后（触发 L2 危险工具即停下等确认）

节点只有两个：think（调 LLM 决策）、act（执行工具）。
其余全是边 —— 这正是"图"相对"循环"的表达力所在。
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import LLMClient
from .loop import (
    STOP_FINISHED,
    STOP_LLM_FAILED,
    STOP_MAX_STEPS,
    STOP_NO_PROGRESS,
    STOP_PENDING_CONFIRM,
    STOP_TOKEN_BUDGET,
    STOP_WALL_CLOCK,
    LoopResult,
    _action_fingerprint,
)
from .observability import emit, span
from .tools import TOOL_SCHEMAS, execute_tool


class LoopState(TypedDict, total=False):
    """图的状态。字段与 loop.py 里的局部变量一一对应，便于逐条核对等价性。"""

    messages: list[dict[str, Any]]
    ctx: dict[str, Any]
    started_at: float
    steps: int
    prompt_tokens: int
    completion_tokens: int
    tools_called: list[str]
    pending: list[dict[str, Any]]
    seen_actions: list[str]
    no_progress: int
    content: str
    stop_reason: str
    degraded: bool
    last_tool_calls: list[dict[str, Any]]


def build_graph(llm: LLMClient, settings: Settings):
    """构建并编译状态图。llm 与 settings 通过闭包注入，避免污染 State。

    图的结构只取决于 llm 与 settings，与具体输入无关，所以下面用 lru_cache
    缓存编译结果 —— 不缓存的话每次请求都要重建一次图，实测会让 P95 明显变差。
    """

    # ----- 节点 1：预算守卫（墙钟 + 步数） -----

    def guard(state: LoopState) -> dict:
        if state.get("stop_reason"):
            return {}
        elapsed = time.monotonic() - state["started_at"]
        if elapsed > settings.wall_clock_budget_s:
            emit("loop_stop", reason=STOP_WALL_CLOCK, elapsed_s=round(elapsed, 2))
            return {"stop_reason": STOP_WALL_CLOCK}
        if state["steps"] >= settings.max_steps:
            emit("loop_stop", reason=STOP_MAX_STEPS)
            return {"stop_reason": STOP_MAX_STEPS}
        return {}

    # ----- 节点 2：Think（调 LLM 决策） -----

    def think(state: LoopState) -> dict:
        step = state["steps"] + 1
        with span("llm_step", step=step):
            resp = llm.chat(state["messages"], tools=TOOL_SCHEMAS)

        prompt_tokens = state["prompt_tokens"] + resp.prompt_tokens
        completion_tokens = state["completion_tokens"] + resp.completion_tokens
        out: dict[str, Any] = {
            "steps": step,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "degraded": state.get("degraded", False) or resp.degraded,
        }

        # 刹车：LLM 全线失败 -> 转人工，而不是返回空答案
        if not resp.ok:
            emit("loop_stop", reason=STOP_LLM_FAILED)
            out["stop_reason"] = STOP_LLM_FAILED
            out["content"] = "抱歉，智能助手暂时不可用，已为您转接人工客服。"
            return out

        # 刹车：token 预算。LLM 一返回就结算，越界立刻停，不再执行本轮工具
        if prompt_tokens + completion_tokens > settings.max_tokens_budget:
            emit(
                "loop_stop",
                reason=STOP_TOKEN_BUDGET,
                total_tokens=prompt_tokens + completion_tokens,
            )
            out["stop_reason"] = STOP_TOKEN_BUDGET
            return out

        # 就地追加 assistant 消息：与 loop.py 保持同一语义（调用方拿到的仍是同一个 list）
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
        if resp.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in resp.tool_calls
            ]
        state["messages"].append(assistant_msg)

        # 正常结束：模型不再请求工具，说明它认为目标已达成
        if not resp.tool_calls:
            emit("loop_stop", reason=STOP_FINISHED, step=step)
            out["stop_reason"] = STOP_FINISHED
            out["content"] = resp.content or ""
            return out

        out["last_tool_calls"] = resp.tool_calls
        return out

    # ----- 节点 3：Act（执行工具 + 观察结果） -----

    def act(state: LoopState) -> dict:
        seen = list(state.get("seen_actions", []))
        no_progress = state.get("no_progress", 0)
        progressed = False
        tools_called = list(state.get("tools_called", []))
        pending = list(state.get("pending", []))

        for tc in state.get("last_tool_calls", []):
            tool_name = tc["name"]
            fp = _action_fingerprint(tool_name, tc["arguments"])

            # 重复动作检测：同样的工具 + 同样的参数再调一次，视为无进展
            if fp in seen:
                no_progress += 1
                emit("loop_duplicate_action", tool=tool_name, step=state["steps"])
            else:
                seen.append(fp)
                no_progress = 0
                progressed = True

            with span("tool_call", step=state["steps"], tool=tool_name):
                tool_result = execute_tool(tool_name, tc["arguments"], state["ctx"])

            tools_called.append(tool_name)
            if tool_result.pending:
                pending.append(tool_result.pending)

            payload = (
                tool_result.payload
                if tool_result.ok
                else {"ok": False, "error": tool_result.error}
            )
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

        out: dict[str, Any] = {
            "seen_actions": seen,
            "no_progress": no_progress,
            "tools_called": tools_called,
            "pending": pending,
        }

        # 刹车：触发 L2 危险工具 -> 停下来等人工确认，不继续自主执行
        if pending:
            emit("loop_stop", reason=STOP_PENDING_CONFIRM, step=state["steps"])
            out["stop_reason"] = STOP_PENDING_CONFIRM
            return out

        # 刹车：连续无进展
        if not progressed and no_progress >= 2:
            emit("loop_stop", reason=STOP_NO_PROGRESS, step=state["steps"])
            out["stop_reason"] = STOP_NO_PROGRESS
            return out

        return out

    # ----- 条件边：刹车全部收在这里，只读 stop_reason，不做副作用 -----

    def after_guard(state: LoopState) -> str:
        return END if state.get("stop_reason") else "think"

    def after_think(state: LoopState) -> str:
        return END if state.get("stop_reason") else "act"

    def after_act(state: LoopState) -> str:
        return END if state.get("stop_reason") else "guard"

    graph = StateGraph(LoopState)
    graph.add_node("guard", guard)
    graph.add_node("think", think)
    graph.add_node("act", act)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", after_guard, {"think": "think", END: END})
    graph.add_conditional_edges("think", after_think, {"act": "act", END: END})
    graph.add_conditional_edges("act", after_act, {"guard": "guard", END: END})

    return graph.compile()


@lru_cache(maxsize=8)
def _compiled_graph(llm: LLMClient, settings: Settings):
    """缓存编译后的图。Settings 是 frozen dataclass（可哈希），LLMClient 用默认对象哈希。"""
    return build_graph(llm, settings)


def run_graph_loop(
    llm: LLMClient,
    messages: list[dict[str, Any]],
    ctx: dict[str, Any],
    settings: Settings,
) -> LoopResult:
    """与 loop.run_loop 签名与返回完全一致，可无缝替换。"""
    started = time.monotonic()
    app = _compiled_graph(llm, settings)

    initial: LoopState = {
        "messages": messages,
        "ctx": ctx,
        "started_at": started,
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tools_called": [],
        "pending": [],
        "seen_actions": [],
        "no_progress": 0,
        "content": "",
        "stop_reason": "",
        "degraded": False,
        "last_tool_calls": [],
    }

    # 每轮最多 3 个节点（guard -> think -> act），留足余量，
    # 避免 LangGraph 自身的递归上限先于我们的步数刹车触发
    final = app.invoke(initial, config={"recursion_limit": settings.max_steps * 4 + 8})

    result = LoopResult(
        content=final.get("content", ""),
        tools_called=final.get("tools_called", []),
        steps=final.get("steps", 0),
        prompt_tokens=final.get("prompt_tokens", 0),
        completion_tokens=final.get("completion_tokens", 0),
        stop_reason=final.get("stop_reason") or STOP_FINISHED,
        pending=final.get("pending", []),
        degraded=final.get("degraded", False),
    )

    # 人工接管：任何异常终止都必须给出明确去向，禁止静默失败
    if result.stop_reason in {
        STOP_MAX_STEPS,
        STOP_TOKEN_BUDGET,
        STOP_WALL_CLOCK,
        STOP_NO_PROGRESS,
    }:
        if not result.content:
            result.content = "这个问题我需要进一步核实，已为您转接人工客服跟进。"

    return result
