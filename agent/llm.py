"""LLM 客户端：显式超时 + 只重试幂等临时性失败 + 主备模型降级 + 成本统计。

三条硬规则（来自错误处理矩阵）：
1. timeout 必须写成显式数值，SDK 默认值往往等于没有超时。
2. 只重试"临时性 + 幂等"的失败。429/5xx/超时可重试；401/400 重试是打爆鉴权或放大错误。
3. 每次降级必须打 warn 日志并计数，否则线上降了级你根本不知道。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import Settings
from .observability import emit


@dataclass
class LLMResult:
    """统一的返回结构：无论成功失败都返回结构化结果，不把异常抛给编排层。"""

    ok: bool
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    error: str = ""
    degraded: bool = False  # 是否走了降级路径


def _is_retryable(exc: BaseException) -> bool:
    """判定可重试：超时/连接错误/429/5xx。其余一律不重试。"""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.RateLimitError):  # 429
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        # max_retries=0：重试逻辑自己用 tenacity 管，避免两层重试叠加成雪崩
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),  # 首次 + 2 次重试
        wait=wait_exponential_jitter(initial=1, max=10),  # 抖动避免同步风暴
        reraise=True,
    )
    def _chat(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, temperature: float) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        # DeepSeek 思考模式默认开启（effort=high）。客服工单场景不需要链式推理，
        # 开启会让输出 token 涨 5~27 倍（按输出价计费，见 tutorials/day12_thinking_400.py）。
        # 关闭方式：extra_body={"thinking": {"type": "disabled"}}，SDK 会在发送前摊平。
        # 注意：直接传 kwargs["thinking"] 会被 OpenAI SDK 拦截（TypeError）；
        # 用 urllib/HTTP 直发时则没有 extra_body 概念，得放请求体顶层。
        if "deepseek" in self.s.llm_base_url.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return self._client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """主入口：先主模型，失败后降级到备用模型，再失败返回结构化错误。"""
        if temperature is None:
            temperature = self.s.llm_temperature
        for attempt, model in enumerate((self.s.llm_model, self.s.llm_fallback_model)):
            try:
                resp = self._chat(model, messages, tools, temperature)
            except RetryError as exc:
                # 重试次数耗尽。emit 后会继续尝试下一个模型。
                emit("llm_retry_exhausted", model=model, error=str(exc)[:200])
                continue
            except openai.OpenAIError as exc:
                # 不可重试类错误（401/400 等）：直接判定终止，不再试备用模型也没意义
                emit("llm_fatal_error", model=model, error=type(exc).__name__)
                return LLMResult(ok=False, error=f"llm_error:{type(exc).__name__}", model=model)

            msg = resp.choices[0].message
            tool_calls = []
            for tc in msg.tool_calls or []:
                tool_calls.append(
                    {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                )
            usage = resp.usage
            result = LLMResult(
                ok=True,
                content=msg.content or "",
                tool_calls=tool_calls,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                model=model,
                degraded=attempt > 0,
            )
            if result.degraded:
                emit("llm_degraded_to_fallback", fallback_model=model)
            emit(
                "llm_call_ok",
                model=model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
            return result

        # 两个模型都失败 —— 降级到底：返回失败，由编排层转成"转人工"
        emit("llm_all_models_failed")
        return LLMResult(ok=False, error="llm_unavailable")

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        on_token=None,
    ) -> LLMResult:
        """流式版 chat：每收到一个 token 片段就回调 on_token，结束后返回完整 LLMResult。

        为什么是「同步流式」而不是 async：
          本项目 LLM 客户端是同步阻塞的。写成 async def 却调同步阻塞代码，
          会把整个事件循环堵死 —— tutorials/day18_async_demo.py 实测过：
          2 个并发从 2.01 秒劣化到 4.00 秒。
          同步生成器交给 StreamingResponse 后由 Starlette 放到线程池迭代，
          既能逐字推送，又不占用事件循环。

        on_token: 可选回调，签名 (delta: str) -> None。
        """
        if temperature is None:
            temperature = self.s.llm_temperature
        for attempt, model in enumerate((self.s.llm_model, self.s.llm_fallback_model)):
            try:
                return self._stream_once(model, messages, tools, temperature, on_token)
            except RetryError as exc:
                emit("llm_retry_exhausted", model=model, error=str(exc)[:200])
                continue
            except openai.OpenAIError as exc:
                emit("llm_fatal_error", model=model, error=type(exc).__name__)
                return LLMResult(ok=False, error=f"llm_error:{type(exc).__name__}", model=model)
        emit("llm_all_models_failed")
        return LLMResult(ok=False, error="llm_unavailable")

    def _stream_once(self, model, messages, tools, temperature, on_token) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},  # 否则拿不到 token 用量，成本观测会失真
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if "deepseek" in self.s.llm_base_url.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        content_parts: list[str] = []
        acc: dict[int, dict[str, str]] = {}
        usage = None
        for chunk in self._client.chat.completions.create(**kwargs):
            # ⚠ 别只在「choices 为空」的 chunk 上找 usage：
            # 实测 DeepSeek 的 usage 出现在**仍带 choices** 的那条上，
            # 只在空 choices 时取会导致 usage 恒为 None、成本观测失效。
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                if on_token is not None:
                    on_token(delta.content)
            # 流式 tool_calls 是按 index 增量下发的，必须自己拼接 arguments
            for tc in delta.tool_calls or []:
                idx = tc.index
                if idx not in acc:
                    acc[idx] = {"id": tc.id or "", "name": tc.function.name or "", "arguments": ""}
                if tc.function is not None and tc.function.arguments:
                    acc[idx]["arguments"] += tc.function.arguments

        tool_calls = [
            {"id": v["id"], "name": v["name"], "arguments": v["arguments"]}
            for _, v in sorted(acc.items())
        ]
        return LLMResult(
            ok=True,
            content="".join(content_parts),
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=model,
            degraded=False,
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """成本估算。单价来自环境变量，默认 0，避免硬编造价格（见 README 待核实清单）。"""
        return (
            prompt_tokens / 1000.0 * self.s.price_per_1k_input
            + completion_tokens / 1000.0 * self.s.price_per_1k_output
        )
