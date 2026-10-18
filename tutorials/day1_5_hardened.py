"""
Day 1.5：把你写的 helloLLM.py 加固成生产级

对照阅读：左边是你原来的写法，右边是生产该有的写法。
这五个差异，就是「demo」和「能上线」之间的全部距离。

用法：
    python day1_5_hardened.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import openai
from openai import OpenAI
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ---------------- 差异 4：结构化日志，而不是 print ----------------
# print 的问题：线上无法按请求聚合，且手机号等敏感信息会原样落盘。
# 关键点：日志写 stderr，不写 stdout。
# 否则日志会和「给用户看的回答」混在同一条流里，终端没法把两者分开
# （比如 `python xxx.py > out.txt` 只想存对话，结果日志也进去了）。
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","event":"%(message)s"}',
    stream=sys.stderr,
)
log = logging.getLogger("hello")

_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")


def mask(text: str) -> str:
    """PII 脱敏：日志里不得出现完整手机号。"""
    return _PHONE_RE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[7:], text)


# ---------------- 差异 1：显式 timeout ----------------
# 你原来没写 timeout。SDK 默认值通常是几百秒（不同版本不同，以官方文档为准）。
# 换算一下：一个请求挂 10 分钟才失败，用户早跑了，连接还占着。
LLM_TIMEOUT_S = 30.0

# ---------------- 差异 2：错误处理 ----------------
# 你原来没做错误处理。网络抖一下、key 过期、余额不足，程序直接崩，
# while True 一崩，整段对话历史全丢。


def is_retryable(exc: BaseException) -> bool:
    """只重试『临时性 + 幂等』的失败。

    429（限流）和 5xx 可重试；401（key 错）重试就是打爆鉴权，绝不能重试。
    这是面试必考题：异常分类错了，比不处理还糟。
    """
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(3),                        # 首次 + 2 次重试
    wait=wait_exponential_jitter(initial=1, max=10),   # 抖动，避免同步风暴
    reraise=True,
)
def _create(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)


def diagnose(exc: Exception) -> str:
    text = str(exc)
    if "401" in text or "invalid_api_key" in text:
        return "API Key 无效或已过期 → 去控制台重新生成"
    if "402" in text or "insufficient" in text.lower():
        return "余额不足 → 充值后再试"
    if "429" in text:
        return "限流 → 降低请求频率，或检查是否余额不足"
    if "Timeout" in type(exc).__name__:
        return "超时 → 检查网络，或调大 LLM_TIMEOUT_S"
    if "Connect" in type(exc).__name__:
        return "连不上 → 检查网络/代理；国内直连 OpenAI 官方域名会失败"
    return f"{type(exc).__name__}: {text[:200]}"


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


# ---------------- 差异 3：history 不能无限增长 ----------------
# 你原来每轮都 append，聊 50 轮后 token 数爆炸：既烧钱，又会撞上上下文窗口上限报错。
# 生产做法：滑动窗口（只保留最近 N 轮）+ 系统提示永不淘汰。
MAX_HISTORY_TURNS = 10  # 保留最近 10 轮问答，即 20 条消息


def trim_history(history: list[dict]) -> list[dict]:
    """保留 system + 最近 MAX_HISTORY_TURNS 轮。

    为什么 system 永不淘汰：它是身份和规则的载体，丢了 Agent 就变了一个人。
    """
    system = [m for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]
    return system + rest[-(MAX_HISTORY_TURNS * 2):]


def ask(client: OpenAI, history: list[dict], model: str) -> tuple[str, int, int]:
    """流式输出，并返回 (完整回复, 输入token, 输出token)。

    差异 5：流式模式下默认拿不到 usage，需要显式要求。
    记不住 token 用量 = 算不出成本 = 面试被问「单次多少钱」时答不上来。
    """
    started = time.perf_counter()
    try:
        stream = _create(
            client=client,
            model=model,
            messages=history,
            stream=True,
            # 让最后一个 chunk 携带 usage。字段名以所用平台文档为准
            stream_options={"include_usage": True},
        )
    except RetryError as exc:
        raise RuntimeError(f"重试 3 次仍失败：{diagnose(exc.last_attempt.exception())}") from exc
    except openai.OpenAIError as exc:
        raise RuntimeError(diagnose(exc)) from exc

    full, prompt_tok, completion_tok = "", 0, 0
    for chunk in stream:
        if getattr(chunk, "usage", None):
            prompt_tok = getattr(chunk.usage, "prompt_tokens", 0) or 0
            completion_tok = getattr(chunk.usage, "completion_tokens", 0) or 0
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
            full += delta
    print()

    cost_ms = int((time.perf_counter() - started) * 1000)
    log.info("llm_call_ok latency_ms=%s prompt_tokens=%s completion_tokens=%s",
             cost_ms, prompt_tok, completion_tok)
    print(f"    〔{cost_ms}ms · 输入 {prompt_tok} / 输出 {completion_tok} token〕")
    return full, prompt_tok, completion_tok


def main() -> int:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    if not api_key:
        print("缺少 API Key。请 cp .env.example .env 并填入 LLM_API_KEY。")
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=LLM_TIMEOUT_S,  # 差异 1
        max_retries=0,          # 重试自己用 tenacity 管，避免两层重试叠加
    )

    history = [{"role": "system", "content": "你是一个有帮助、简洁、友好的 AI 助手。"}]
    total_prompt = total_completion = 0

    print("输入 exit / quit 退出。")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if question.lower() in {"exit", "quit", "退出"}:
            print(f"已退出。本次累计：输入 {total_prompt} / 输出 {total_completion} token")
            break
        if not question:
            continue

        log.info("user_input %s", json.dumps(mask(question), ensure_ascii=False))
        history.append({"role": "user", "content": question})
        history = trim_history(history)  # 差异 3

        print("AI：", end="", flush=True)
        try:
            reply, p_tok, c_tok = ask(client, history, model)
        except RuntimeError as exc:
            print(f"\n出错了：{exc}")
            # 关键：出错不退出循环，把刚加的用户消息撤掉，让用户能继续
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        total_prompt += p_tok
        total_completion += c_tok

    return 0


if __name__ == "__main__":
    sys.exit(main())
