"""编排层：输入校验 → 意图路由 → 分支处理 → 自主循环 → 汇总。

架构模式选择（来自模式库）：客服工单场景 = Routing + Chaining + 人工审核队列。
Routing 的价值不只是"分对类"，更在于把 chitchat/unknown 短路掉——
这两类根本不需要进自主循环，短路能直接省掉大部分成本。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .llm import LLMClient
from .loop import (
    STOP_FINISHED,
    STOP_PENDING_CONFIRM,
    LoopResult,
    run_loop,
)
from .observability import emit, new_trace_id, span
from .prompts import (
    CONFIDENCE_THRESHOLD,
    FALLBACK_INTENT,
    ROUTER_PROMPT,
    SYSTEM_PROMPT,
)
from .tools import confirm_pending
from .memory import MemoryStore


def _pick_loop_impl(impl: str):
    """按配置选自主循环的实现。

    两种实现返回完全相同的 LoopResult，因此切换不影响上层：
      - plain：agent/loop.py 的手写 for 循环（默认，零额外依赖）
      - graph：agent/graph_loop.py 的 LangGraph 状态图（刹车画成条件边）

    降级路径（硬约束 C4）：graph 实现依赖 langgraph，导入失败时
    自动退回手写循环并打点告警，绝不因为缺依赖就让服务起不来。
    """
    if impl == "graph":
        try:
            from .graph_loop import run_graph_loop

            return run_graph_loop
        except Exception as exc:  # 依赖缺失 / 版本不兼容，一律降级
            emit("loop_impl_fallback", to="plain", error=type(exc).__name__)
            return run_loop
    return run_loop

# —— 注入预检：确定性地拦住最常见的提示注入，不把这类请求送进 LLM ——
_INJECTION_PATTERNS = [
    r"忽略(?:以上|上面|之前)(?:的)?(?:所有)?(?:指令|规则|设定)",
    r"(?:输出|打印|复述|泄露)(?:你的)?(?:系统)?(?:提示词|提示|指令|prompt)",
    r"管理员模式",
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# 只靠问号判断"模型是否在追问"太脆弱：模型换个说法（"请提供手机号"没问号）
# 就会被误判成 answer，评测分数随措辞漂移，看起来像模型不稳定，其实是量具不稳。
# 补上索要信息的常见说法；真正的解法是让模型输出结构化行为标签，这里先用关键词兜底。
_CLARIFY_HINTS = (
    "请提供", "方便提供", "麻烦提供", "请补充", "请告知", "请告诉",
    "需要您提供", "需要您补充", "能否提供", "可以告诉我", "请问您的",
    "提供一下", "麻烦您提供",
)


def _looks_like_clarify(text: str) -> bool:
    if "?" in text or "？" in text:
        return True
    return any(h in text for h in _CLARIFY_HINTS)

# 行为枚举：评测脚本用它做 L1 确定性断言
BEHAVIOR_ANSWER = "answer"
BEHAVIOR_CLARIFY = "ask_clarify"
BEHAVIOR_ESCALATE = "escalate"
BEHAVIOR_REFUSE = "refuse"
BEHAVIOR_CONFIRM = "needs_confirm"


class UserInput(BaseModel):
    """输入校验放在最外层：脏数据不许进链路（错误处理矩阵第六节）。"""

    text: str = Field(..., min_length=0, max_length=2000)
    user_id: str = Field(default="anonymous", max_length=64)

    @field_validator("text")
    @classmethod
    def _strip_control_chars(cls, v: str) -> str:
        # 去掉控制字符（含零宽字符），保留换行与制表符
        cleaned = "".join(ch for ch in v if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
        return cleaned.strip()


@dataclass
class AgentResponse:
    behavior: str
    output: str
    trace_id: str
    tools_called: list[str] = field(default_factory=list)
    steps: int = 0
    stop_reason: str = ""
    pending: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    cost: float = 0.0
    intent: str = ""
    degraded: bool = False


class TicketAgent:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.llm = LLMClient(settings)
        self.memory = MemoryStore(self.s.memory_db_path)  # 长期记忆：跨会话记住每个用户
        self._loop = _pick_loop_impl(settings.loop_impl)  # 自主循环实现（可切换）

    # ---------------- 路由 ----------------

    def _route(self, text: str) -> str:
        """意图分类。失败或非合法 JSON 一律走兜底 unknown —— 禁止默认归到第一类。"""
        prompt = ROUTER_PROMPT.format(user_input=text[:500])
        messages = [
            {"role": "system", "content": "你是严格的意图分类器。"},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.llm.chat(messages, tools=None, temperature=0.0)
        except Exception as exc:
            emit("router_error", error=type(exc).__name__)
            return FALLBACK_INTENT

        if not resp.ok:
            return FALLBACK_INTENT

        raw = resp.content.strip()
        # 模型偶尔会包 ```json 围栏，先剥离再解析
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(raw)
            intent = parsed.get("intent", FALLBACK_INTENT)
            confidence = float(parsed.get("confidence", 0.0))
        except (json.JSONDecodeError, ValueError, AttributeError):
            emit("router_bad_json", raw_preview=raw[:80])
            return FALLBACK_INTENT

        # 置信度阈值 + 枚举校验，两道闸
        if intent not in {"faq", "query", "action", "chitchat", "unknown"}:
            return FALLBACK_INTENT
        if confidence < CONFIDENCE_THRESHOLD:
            return FALLBACK_INTENT
        return intent

    # ---------------- 主流程 ----------------

    def handle(self, text: str, user_id: str = "anonymous") -> AgentResponse:
        started = time.perf_counter()
        trace_id = new_trace_id()
        emit("request_start", user_id=user_id, input_len=len(text))

        try:
            user_input = UserInput(text=text, user_id=user_id)
        except Exception as exc:
            # 输入校验失败 = 终止类错误，直接拒，不浪费一次 LLM 调用
            emit("input_rejected", error=type(exc).__name__)
            return AgentResponse(
                behavior=BEHAVIOR_REFUSE,
                output="您输入的内容超出处理范围（长度上限 2000 字），请精简后重试。",
                trace_id=trace_id,
            )

        clean = user_input.text

        # 空输入：不做任何 LLM 调用，直接追问（边界用例 B01）
        if not clean:
            return AgentResponse(
                behavior=BEHAVIOR_CLARIFY,
                output="您好，请描述一下您遇到的问题，例如：查询订单、申请退货、咨询退款时效。",
                trace_id=trace_id,
            )

        # 注入预检：确定性拒绝，不进 LLM（对抗用例 A01/A05）
        for pattern in _INJECTION_RE:
            if pattern.search(clean):
                emit("injection_blocked", pattern=pattern.pattern[:40])
                return AgentResponse(
                    behavior=BEHAVIOR_REFUSE,
                    output="抱歉，该请求涉及获取系统内部设定，我无法执行。如需售后帮助，请描述您的订单问题。",
                    trace_id=trace_id,
                )

        with span("route"):
            intent = self._route(clean)

        # 短路分支：不进自主循环，省成本也降风险
        if intent == "chitchat":
            return AgentResponse(
                behavior=BEHAVIOR_ANSWER,
                output="您好，我是售后智能助手，可以帮您查询订单、物流，以及办理退货退款。请问有什么可以帮您？",
                trace_id=trace_id,
                intent=intent,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        if intent == "unknown":
            with span("escalate"):
                emit("escalate_to_human", reason="intent_unknown")
            return AgentResponse(
                behavior=BEHAVIOR_ESCALATE,
                output="没能准确理解您的需求，已为您转接人工客服，请稍候。",
                trace_id=trace_id,
                intent=intent,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        # 长期记忆注入：把该用户最近的对话历史拼到 system 之后、当前问题之前。
        # 历史为空（新用户 / 评测独立 user_id）时，等价于改造前的 [system, user]，行为不变。
        history = self.memory.load_history(user_input.user_id, limit=6)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": clean})
        ctx = {"user_id": user_input.user_id, "trace_id": trace_id}

        with span("agent_loop", intent=intent):
            loop_result: LoopResult = self._loop(self.llm, messages, ctx, self.s)

        # 长期记忆落盘：把这一轮（用户问题 + 最终回答）存回，供后续会话回忆
        self.memory.save_turn(user_input.user_id, clean, loop_result.content)

        behavior = self._to_behavior(loop_result, clean)
        cost = self.llm.estimate_cost(loop_result.prompt_tokens, loop_result.completion_tokens)

        emit(
            "request_end",
            intent=intent,
            behavior=behavior,
            steps=loop_result.steps,
            stop_reason=loop_result.stop_reason,
            tools=loop_result.tools_called,
            cost=round(cost, 6),
        )

        return AgentResponse(
            behavior=behavior,
            output=loop_result.content,
            trace_id=trace_id,
            tools_called=loop_result.tools_called,
            steps=loop_result.steps,
            stop_reason=loop_result.stop_reason,
            pending=loop_result.pending,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cost=cost,
            intent=intent,
            degraded=loop_result.degraded,
        )

    @staticmethod
    def _to_behavior(loop_result: LoopResult, user_text: str) -> str:
        """把循环的内部状态翻译成对外行为标签，供评测与监控使用。"""
        if loop_result.stop_reason == STOP_PENDING_CONFIRM:
            return BEHAVIOR_CONFIRM
        if "escalate_to_human" in loop_result.tools_called:
            return BEHAVIOR_ESCALATE
        if loop_result.stop_reason != STOP_FINISHED:
            return BEHAVIOR_ESCALATE
        # 模型主动追问（例如缺手机号/订单号）
        if not loop_result.tools_called and _looks_like_clarify(loop_result.content):
            return BEHAVIOR_CLARIFY
        return BEHAVIOR_ANSWER

    # ---------------- 人工确认 L2 动作 ----------------

    def confirm(self, pending: dict[str, Any], user_id: str = "anonymous") -> dict[str, Any]:
        """L2 危险工具的人工确认入口。幂等键保证重复确认不会重复建单。"""
        emit("human_confirm_received", idempotency_key=pending.get("idempotency_key", ""))
        return confirm_pending(pending, {"user_id": user_id})
