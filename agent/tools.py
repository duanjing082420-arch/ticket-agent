"""工具层：Agent 改变外部世界的唯一手段。

设计原则（来自工具设计规范）：
- 一个工具一件事，动词开头，名字即文档
- 参数最小化 + pattern/minimum/maximum 双重约束
- 危险工具（L2）只生成"待确认动作"，绝不直接执行
- 失败返回结构化错误，不抛异常到编排层（否则整条链路会崩）
- 结果一律截断，防止巨型返回撑爆上下文
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .observability import emit, mask_pii

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "mock_orders.json"
MAX_RESULT_CHARS = 3000

# 权限分级：L0 只读 / L1 可撤销写 / L2 不可逆写（必须人工确认）
RISK_LEVEL: dict[str, int] = {
    "search_orders": 0,
    "get_shipment": 0,
    "search_policy": 0,
    "escalate_to_human": 1,
    "create_ticket": 2,  # 会写入工单系统并可能触发对外通知
}


def _load_data() -> dict[str, Any]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """截断优于报错：报错会让整条链路崩掉。"""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[已截断 {len(text) - limit} 字符]"


# --------------------------------------------------------------------------
# 工具 JSON Schema（OpenAI function calling 格式）
# description 写"何时用"，不写"怎么用"——模型要判断的是调用时机
# --------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "按手机号或订单号查询订单。仅当用户已提供手机号或订单号时才调用；两者都没提供时不要调用，先向用户追问。",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "description": "11 位中国大陆手机号。与 order_id 二选一。",
                        "pattern": "^1[3-9]\\d{9}$",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20240917001。与 phone 二选一。",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回条数，1-20。",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipment",
            "description": "查询指定订单的物流轨迹与承运商信息。仅在已知订单号且用户关心发货/物流进度时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20240917001。",
                        "pattern": "^SO\\d{11}$",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "检索售后政策知识库（退货、退款时效、发货时效、物流异常、质保）。仅当用户询问规则、时效、能否退货退款时调用；不要用它查订单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "检索关键词，如：退货 / 退款 / 发货 / 物流 / 质保。",
                        "minLength": 1,
                        "maxLength": 20,
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "转人工客服。仅当信息不足、用户情绪激烈、涉及赔付/投诉、或多次尝试仍无法解决时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "转人工原因，一句话说明用户诉求与已尝试的动作。",
                        "minLength": 5,
                        "maxLength": 200,
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "创建售后工单（退货/退款/物流异常/投诉）。这是不可逆写操作，调用后会生成待确认动作，需要人工确认后才会真正执行。禁止在用户未明确要求处理时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20240917001。",
                        "pattern": "^SO\\d{11}$",
                    },
                    "category": {
                        "type": "string",
                        "description": "工单类型。",
                        "enum": ["退货", "退款", "物流异常", "投诉", "质保维修"],
                    },
                    "description": {
                        "type": "string",
                        "description": "问题描述，需包含用户诉求与关键信息。",
                        "minLength": 5,
                        "maxLength": 500,
                    },
                },
                "required": ["order_id", "category", "description"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class ToolResult:
    name: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    risk_level: int = 0
    pending: dict[str, Any] | None = None  # L2 工具的待确认动作


def make_idempotency_key(user_id: str, tool: str, params: dict[str, Any], bucket_s: int = 300) -> str:
    """幂等键：限定时间窗口，避免永久阻断合理重试。"""
    bucket = int(time.time()) // bucket_s
    raw = f"{user_id}:{tool}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}:{bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 工具实现
# --------------------------------------------------------------------------


def _search_orders(args: dict[str, Any], ctx: dict[str, Any]) -> ToolResult:
    phone = args.get("phone")
    order_id = args.get("order_id")
    limit = int(args.get("limit", 5) or 5)

    if not phone and not order_id:
        # 参数缺失：返回具体原因 + 正确格式，让模型能自我修正而不是干瞪眼
        return ToolResult(
            "search_orders", False, error="missing_identifier: 需要提供 phone 或 order_id 之一"
        )
    if phone and not re.match(r"^1[3-9]\d{9}$", phone):
        return ToolResult("search_orders", False, error="bad_phone: 手机号需为 11 位大陆号码")
    if limit < 1 or limit > 20:
        return ToolResult("search_orders", False, error="bad_limit: limit 需在 1-20 之间")

    orders = _load_data()["orders"]
    hits = [
        o
        for o in orders
        if (order_id and o["order_id"] == order_id) or (phone and o["phone"] == phone)
    ][:limit]

    # 空结果是合法答案，不要返回错误
    emit("tool_search_orders", hit_count=len(hits))
    slim = [
        {
            "order_id": o["order_id"],
            "status": o["status"],
            "item": o["item"],
            "amount": o["amount"],
            "paid_at": o["paid_at"],
        }
        for o in hits
    ]
    return ToolResult("search_orders", True, payload={"orders": slim, "count": len(slim)})


def _get_shipment(args: dict[str, Any], ctx: dict[str, Any]) -> ToolResult:
    order_id = args.get("order_id", "")
    orders = _load_data()["orders"]
    hit = next((o for o in orders if o["order_id"] == order_id), None)
    if not hit:
        return ToolResult("get_shipment", True, payload={"found": False, "message": "未找到该订单"})
    payload = {
        "found": True,
        "order_id": hit["order_id"],
        "status": hit["status"],
        "carrier": hit["carrier"],
        "tracking_no": hit["tracking_no"],
        "shipped_at": hit["shipped_at"],
        # 真实场景这里会调承运商 API；本例用 mock 轨迹
        "traces": [
            "2024-09-18 09:05 已发货，承运商揽收",
            "2024-09-18 18:32 到达【杭州转运中心】",
            "2024-09-19 07:41 派送中",
        ]
        if hit["shipped_at"]
        else [],
    }
    emit("tool_get_shipment", order_id=order_id)
    return ToolResult("get_shipment", True, payload=payload)


def _search_policy(args: dict[str, Any], ctx: dict[str, Any]) -> ToolResult:
    keyword = (args.get("keyword") or args.get("query") or "").strip()
    if not keyword:
        return ToolResult("search_policy", False, error="missing_keyword")
    try:
        # 真检索：语义向量检索 data/policies.md（15 条真实售后政策）
        # 把"意思相近"的政策找出来，而不是死板地找"相同的字"
        from .rag import retrieve
        hits = retrieve(keyword, top_k=3)
    except Exception:
        # 兜底降级：embedding 服务异常 / 无 key 时，退回关键词匹配。
        # 目的不是结果好，而是"检索链路挂了也不能让整个工具崩"。
        policies = _load_data()["policies"]
        hits = [p for p in policies if keyword in p["title"] or keyword in p["body"]]
        if not hits:
            hits = [p for p in policies if any(ch in p["body"] for ch in keyword)]
        hits = [{"id": p["id"], "title": p["title"], "body": p["body"]} for p in hits]
    emit("tool_search_policy", keyword=keyword, hit_count=len(hits), retriever="rag" if len(hits) <= 3 else "fallback")
    return ToolResult(
        "search_policy",
        True,
        payload={"policies": hits},
    )


def _escalate_to_human(args: dict[str, Any], ctx: dict[str, Any]) -> ToolResult:
    reason = (args.get("reason") or "").strip()
    if len(reason) < 5:
        return ToolResult("escalate_to_human", False, error="bad_reason: 原因至少 5 个字")
    emit("tool_escalate", reason_preview=mask_pii(reason[:50]))
    return ToolResult(
        "escalate_to_human", True, payload={"queued": True, "reason": reason}, risk_level=1
    )


def _create_ticket(args: dict[str, Any], ctx: dict[str, Any]) -> ToolResult:
    """L2 危险工具：只生成待确认动作，不真正执行。"""
    order_id = args.get("order_id", "")
    category = args.get("category", "")
    description = (args.get("description") or "").strip()

    if not re.match(r"^SO\d{11}$", order_id):
        return ToolResult("create_ticket", False, error="bad_order_id: 订单号格式应为 SO + 11 位数字")
    if category not in {"退货", "退款", "物流异常", "投诉", "质保维修"}:
        return ToolResult("create_ticket", False, error="bad_category: category 取值不在枚举内")
    if len(description) < 5:
        return ToolResult("create_ticket", False, error="bad_description: 描述至少 5 个字")

    key = make_idempotency_key(ctx.get("user_id", "anon"), "create_ticket", args)
    preview = f"将创建工单：订单 {order_id} / 类型 {category} / 描述 {description[:60]}"
    emit("tool_create_ticket_pending", idempotency_key=key)
    return ToolResult(
        "create_ticket",
        True,
        payload={"message": "已生成待确认工单，等待人工确认后执行"},
        risk_level=2,
        pending={"tool": "create_ticket", "args": args, "preview": preview, "idempotency_key": key},
    )


_IMPL: dict[str, Callable[[dict[str, Any], dict[str, Any]], ToolResult]] = {
    "search_orders": _search_orders,
    "get_shipment": _get_shipment,
    "search_policy": _search_policy,
    "escalate_to_human": _escalate_to_human,
    "create_ticket": _create_ticket,
}

# 已执行的幂等键 → 结果，避免重复提交
_EXECUTED: dict[str, dict[str, Any]] = {}


def execute_tool(name: str, raw_args: str | dict[str, Any], ctx: dict[str, Any]) -> ToolResult:
    """统一入口：解析参数 → 校验 → 执行 → 结构化返回。任何异常都收敛成 ok=False。"""
    if name not in _IMPL:
        return ToolResult(name, False, error="unknown_tool")

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            # 模型输出的 arguments 不是合法 JSON 是常见故障，返回明确错误让其重试
            return ToolResult(name, False, error="bad_json_arguments")
    else:
        args = raw_args

    try:
        result = _IMPL[name](args, ctx)
    except Exception as exc:  # 兜底：工具内部异常不得击穿编排层
        emit("tool_exception", tool=name, error=type(exc).__name__)
        return ToolResult(name, False, error=f"tool_exception:{type(exc).__name__}")

    # 统一截断结果，防止撑爆上下文
    if result.ok and result.payload:
        text = json.dumps(result.payload, ensure_ascii=False)
        if len(text) > MAX_RESULT_CHARS:
            result.payload = {"summary": _truncate(text), "truncated": True}
    return result


def confirm_pending(pending: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """人工确认后真正执行 L2 动作。幂等键命中则直接返回上次结果。"""
    key = pending["idempotency_key"]
    if key in _EXECUTED:
        emit("tool_idempotent_hit", idempotency_key=key)
        return {"executed": False, "reason": "幂等键命中，未重复执行", "result": _EXECUTED[key]}

    args = pending["args"]
    ticket_id = "TK" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:10].upper()
    record = {
        "ticket_id": ticket_id,
        "order_id": args["order_id"],
        "category": args["category"],
        "description": args["description"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _EXECUTED[key] = record
    emit("tool_create_ticket_executed", ticket_id=ticket_id, idempotency_key=key)
    return {"executed": True, "result": record}
