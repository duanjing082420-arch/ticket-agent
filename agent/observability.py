"""可观测层：trace_id 贯穿 + 结构化日志 + PII 脱敏。

为什么不用 print：线上出问题时你需要按 trace_id 把一次请求的全部 span 串起来，
print 做不到，且会把手机号等 PII 原样写进日志（合规风险）。
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

# trace_id 用 contextvars 而非全局变量：异步/并发下不会串号。
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

# 11 位大陆手机号
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
# 简单邮箱
_EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    trace_id_var.set(tid)
    return tid


def mask_pii(text: str) -> str:
    """脱敏：手机号中间四位打星。日志与 trace 里不得出现原始 PII。"""
    if not text:
        return text
    text = _PHONE_RE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[7:], text)
    text = _EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@" + m.group(0).split("@", 1)[1], text)
    return text


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志，方便丢进 ELK / Loki 做检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "event": record.getMessage(),
        }
        # 额外字段（span 埋点数据）合并进来并统一脱敏
        for key, val in getattr(record, "extra_fields", {}).items():
            payload[key] = mask_pii(val) if isinstance(val, str) else val
        return json.dumps(payload, ensure_ascii=False)


def force_utf8_stdout() -> None:
    """把 stdout/stderr 固定成 UTF-8。

    Windows 上 Python 一旦发现输出被重定向或接了管道，就会退回系统 ANSI 编码（GBK），
    此时打印 ❌ ✅ 这类字符会抛 UnicodeEncodeError。坑在于它往往崩在最后一行的汇总：
    报告文件已经落盘，但退出码变成 1，CI 会误判成失败，人肉看输出又像是"跑完了"。
    所以每个 CLI 入口都要在最前面显式固定编码，且必须早于 setup_logging。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


log = logging.getLogger("ticket_agent")


def emit(event: str, **fields: Any) -> None:
    """结构化埋点。字段值若是字符串会被自动脱敏。"""
    safe = {k: (mask_pii(v) if isinstance(v, str) else v) for k, v in fields.items()}
    log.info(event, extra={"extra_fields": safe})


@contextmanager
def span(name: str, **fields: Any) -> Iterator[None]:
    """span 埋点：记录每个节点的耗时与成败。失败也会打点（不会静默）。"""
    start = time.perf_counter()
    emit(f"span.start::{name}", **fields)
    try:
        yield
    except Exception as exc:  # 向上抛，由调用方决定降级/重试，但这里必须留下痕迹
        emit(f"span.error::{name}", error=type(exc).__name__, **fields)
        raise
    finally:
        cost_ms = int((time.perf_counter() - start) * 1000)
        emit(f"span.end::{name}", latency_ms=cost_ms, **fields)
