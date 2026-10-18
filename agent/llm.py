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
        return self._client.chat.completions.create(**kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """主入口：先主模型，失败后降级到备用模型，再失败返回结构化错误。"""
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

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """成本估算。单价来自环境变量，默认 0，避免硬编造价格（见 README 待核实清单）。"""
        return (
            prompt_tokens / 1000.0 * self.s.price_per_1k_input
            + completion_tokens / 1000.0 * self.s.price_per_1k_output
        )
