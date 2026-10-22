"""自主循环（Autonomous Loop / ReAct）：Think → Act → Observe → 循环。

硬约束 C5 四件套，一个都不能少：
1. max_steps      —— 步数上限
2. token 预算     —— 成本上限
3. 墙钟上限       —— 时间上限
4. 停止条件       —— 目标达成 / 重复动作 / 无进展 / 预算耗尽
5. 人工接管触发   —— 任一异常终止都转成"转人工"而不是静默返回

禁止 while True：下面的 for 循环有明确上界，这是与 demo 的分界线。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .llm import LLMClient, LLMResult
from .observability import emit, span
from .tools import TOOL_SCHEMAS, execute_tool

# 停止原因枚举，便于在日志与评测里归因
STOP_FINISHED = "finished"            # 模型不再请求工具，正常结束
STOP_MAX_STEPS = "max_steps_reached"
STOP_TOKEN_BUDGET = "token_budget_exceeded"
STOP_WALL_CLOCK = "wall_clock_exceeded"
STOP_NO_PROGRESS = "no_progress"
STOP_LLM_FAILED = "llm_unavailable"
STOP_PENDING_CONFIRM = "pending_human_confirm"


@dataclass
class LoopResult:
    content: str
    tools_called: list[str] = field(default_factory=list)
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = STOP_FINISHED
    pending: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False


def _action_fingerprint(name: str, args: str) -> str:
    return hashlib.md5(f"{name}:{args}".encode("utf-8")).hexdigest()


def run_loop(
    llm: LLMClient,
    messages: list[dict[str, Any]],
    ctx: dict[str, Any],
    settings: Settings,
    on_token=None,
) -> LoopResult:
    """执行有界自主循环。messages 会被就地追加（调用方如需保留可自行深拷贝）。

    on_token: 可选回调 (delta: str) -> None。传了就走流式 LLM 调用，
    生成的每个 token 片段都会回调一次 —— 供 SSE 端点逐字推给前端。
    不传则走普通的 llm.chat()，行为与改造前完全一致（保证评测不受影响）。
    """
    started = time.monotonic()
    result = LoopResult(content="")

    seen_actions: set[str] = set()
    consecutive_no_progress = 0

    for step in range(1, settings.max_steps + 1):
        # —— 墙钟检查：放在循环开头，保证任何情况下都不会超时运行 ——
        elapsed = time.monotonic() - started
        if elapsed > settings.wall_clock_budget_s:
            result.stop_reason = STOP_WALL_CLOCK
            emit("loop_stop", reason=STOP_WALL_CLOCK, elapsed_s=round(elapsed, 2))
            break

        with span("llm_step", step=step):
            # 只有传了 on_token 才走流式；否则保持原路径，确保非流式场景零变化
            if on_token is not None:
                resp: LLMResult = llm.stream_chat(messages, tools=TOOL_SCHEMAS, on_token=on_token)
            else:
                resp: LLMResult = llm.chat(messages, tools=TOOL_SCHEMAS)

        result.steps = step
        result.prompt_tokens += resp.prompt_tokens
        result.completion_tokens += resp.completion_tokens
        result.degraded = result.degraded or resp.degraded

        if not resp.ok:
            # LLM 全线失败：降级路径是"转人工"，不是返回空答案
            result.stop_reason = STOP_LLM_FAILED
            result.content = "抱歉，智能助手暂时不可用，已为您转接人工客服。"
            emit("loop_stop", reason=STOP_LLM_FAILED)
            break

        # —— token 预算检查 ——
        total_tokens = result.prompt_tokens + result.completion_tokens
        if total_tokens > settings.max_tokens_budget:
            result.stop_reason = STOP_TOKEN_BUDGET
            emit("loop_stop", reason=STOP_TOKEN_BUDGET, total_tokens=total_tokens)
            break

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
        messages.append(assistant_msg)

        # —— 停止条件：模型不再请求工具，说明它认为目标已达成 ——
        if not resp.tool_calls:
            result.content = resp.content or ""
            result.stop_reason = STOP_FINISHED
            emit("loop_stop", reason=STOP_FINISHED, step=step)
            break

        # —— Act：执行工具；Observe：把结果写回消息列表 ——
        progressed = False
        for tc in resp.tool_calls:
            tool_name = tc["name"]
            fp = _action_fingerprint(tool_name, tc["arguments"])

            # 重复动作检测：同样的工具+同样参数再调一次，视为无进展
            if fp in seen_actions:
                consecutive_no_progress += 1
                emit("loop_duplicate_action", tool=tool_name, step=step)
            else:
                seen_actions.add(fp)
                consecutive_no_progress = 0
                progressed = True

            with span("tool_call", step=step, tool=tool_name):
                tool_result = execute_tool(tool_name, tc["arguments"], ctx)

            result.tools_called.append(tool_name)

            if tool_result.pending:
                result.pending.append(tool_result.pending)

            payload = (
                tool_result.payload
                if tool_result.ok
                else {"ok": False, "error": tool_result.error}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

        if not progressed and consecutive_no_progress >= 2:
            result.stop_reason = STOP_NO_PROGRESS
            emit("loop_stop", reason=STOP_NO_PROGRESS, step=step)
            break

        if result.pending:
            # 触发了 L2 危险工具：停下来等人工确认，不继续自主执行
            result.stop_reason = STOP_PENDING_CONFIRM
            emit("loop_stop", reason=STOP_PENDING_CONFIRM, step=step)
            break
    else:
        # for-else：循环正常跑完所有步数仍未结束 —— 达到步数上限
        result.stop_reason = STOP_MAX_STEPS
        emit("loop_stop", reason=STOP_MAX_STEPS)

    # —— 人工接管：任何异常终止都必须给出明确去向，禁止静默失败 ——
    if result.stop_reason in {
        STOP_MAX_STEPS,
        STOP_TOKEN_BUDGET,
        STOP_WALL_CLOCK,
        STOP_NO_PROGRESS,
    }:
        if not result.content:
            result.content = "这个问题我需要进一步核实，已为您转接人工客服跟进。"

    return result
