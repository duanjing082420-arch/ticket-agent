"""
Day 1：第一次调用大模型（最小可运行示例）

目标：亲眼确认「一次 LLM 调用 = 一次 HTTP POST」，看清楚请求体和响应体长什么样。
这一步没打通，后面所有代码都是空中楼阁。

用法：
    python day1_hello_llm.py            真正调用（需要 .env 里有 key）
    python day1_hello_llm.py --dry-run  不联网，只打印将要发送的请求体
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    """极简 .env 加载器。

    为什么不用 python-dotenv：今天的目标是理解原理，多一个依赖就多一层黑盒。
    等你自己会写了，再换成库不迟。
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        # setdefault：命令行里已设置的环境变量优先级更高，不被 .env 覆盖
        os.environ.setdefault(key.strip(), val.strip())


def build_messages(question: str) -> list[dict[str, str]]:
    """对话 = 一个角色数组。这是与模型交互的唯一入口。

    role 有三种：
      system    —— 设定身份和规则，优先级最高
      user      —— 用户说的话
      assistant —— 模型之前说过的话（多轮对话时用来带上下文）
    """
    return [
        {"role": "system", "content": "你是一名简洁的技术助手，回答不超过两句话。"},
        {"role": "user", "content": question},
    ]


def diagnose(exc: Exception) -> str:
    """把常见报错翻译成人话。

    排查顺序是有讲究的：先看凭据，再看地址，最后看网络。
    反过来查会浪费大量时间。
    """
    name = type(exc).__name__
    text = str(exc)
    if "401" in text or "Authentication" in text or "invalid_api_key" in text:
        return "API Key 不对或已失效 → 去厂商控制台重新生成，确认 .env 里没有多余空格或引号"
    if "404" in text or "model" in text.lower() and "not found" in text.lower():
        return "模型名或 base_url 不对 → 核对厂商文档里的模型名，注意 base_url 末尾是否多/少了 /v1"
    if "429" in text or "rate" in text.lower():
        return "被限流了 → 通常是余额不足或 QPS 超限，去控制台看余额"
    if "Timeout" in name or "timed out" in text:
        return "请求超时 → 检查网络能否访问该域名，或把超时调大到 60 秒再试"
    if "Connect" in name or "Connection" in text:
        return "连不上 → 检查网络、代理、防火墙；国内直连 OpenAI 官方域名会失败，请换国内厂商"
    return f"{name}: {text[:300]}"


def main() -> int:
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    # 从命令行取问题，默认问一个固定问题，方便你对照输出
    question = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "用一句话解释什么是 AI Agent"

    messages = build_messages(question)

    payload_preview = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    print("=" * 60)
    print("即将发送的请求体（这就是全部秘密，没有别的了）")
    print("=" * 60)
    print(json.dumps(payload_preview, ensure_ascii=False, indent=2))

    if "--dry-run" in sys.argv:
        print("\n[dry-run 模式] 未真正发送请求。")
        print(f"目标地址：{base_url}")
        print(f"API Key 已配置：{'是' if api_key else '否（请先填 .env）'}")
        return 0

    if not api_key:
        print("\n缺少 LLM_API_KEY。请执行：cp .env.example .env 并填入你的 key。")
        return 2

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,   # 必须显式写。SDK 默认值往往等于没有超时，生产上这是事故源头
        max_retries=0,  # 自己管重试，避免和 SDK 内置重试叠成雪崩
    )

    print(f"\n正在调用 {model} ...")
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except Exception as exc:
        cost = int((time.perf_counter() - started) * 1000)
        print(f"\n调用失败（耗时 {cost}ms）")
        print(f"诊断：{diagnose(exc)}")
        return 1

    cost_ms = int((time.perf_counter() - started) * 1000)
    content = resp.choices[0].message.content or ""
    usage = resp.usage

    print("=" * 60)
    print("模型回复")
    print("=" * 60)
    print(content)

    print("\n" + "=" * 60)
    print("本次调用的账（面试必问的成本意识就从这里开始）")
    print("=" * 60)
    print(f"耗时        : {cost_ms} ms")
    print(f"输入 token  : {getattr(usage, 'prompt_tokens', 0)}")
    print(f"输出 token  : {getattr(usage, 'completion_tokens', 0)}")
    print(f"合计 token  : {getattr(usage, 'total_tokens', 0)}")
    print(f"finish_reason: {resp.choices[0].finish_reason}")

    print("\n" + "=" * 60)
    print("今天的结论")
    print("=" * 60)
    print("1. 你刚才做的事 = 发一个 HTTP POST，收一个 JSON。就这些。")
    print("2. 模型不知道你的订单、不知道你的数据库、不知道今天几号。")
    print("   它只有你发给它的 messages —— 明天我们就从这个缺陷出发。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
