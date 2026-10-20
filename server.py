"""FastAPI 服务化：把命令行 Agent 暴露成 HTTP 接口。

启动：
    uvicorn server:app --reload --port 8000
之后打开 http://127.0.0.1:8000/docs 看自动生成的接口文档。

设计要点 —— 为什么要拆成 /chat 和 /confirm 两个端点：
CLI（app.py）里 L2 危险动作是这样卡住的：
    answer = input("是否确认执行？(y/N)：")   # 阻塞，等用户敲键盘
HTTP 服务端无法阻塞等待用户输入。所以拆成两次请求：
    POST /chat    -> 返回待确认动作（pending，含幂等键），behavior=needs_confirm，不真正执行
    POST /confirm -> 客户端把 pending 原样回传，用户点了「确认」才真正执行
即「把一次有状态的人机交互拆成两次无状态请求」。配合幂等键，
保证同一条 pending 重复提交不会重复建单（与 CLI 的 y/N 完全一致的安全等级）。

为什么路由函数是 def 而不是 async def：
FastAPI 会把普通 def 放进线程池执行，因此 agent.handle() 里 2~5 秒的 LLM 阻塞调用
不会卡住事件循环；若写成 async def 却调用同步阻塞代码，反而会阻塞整个服务。
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent.config import Settings, load_dotenv  # noqa: E402
from agent.observability import setup_logging  # noqa: E402
from agent.service import TicketAgent  # noqa: E402

# 单例：进程启动时建一次 Agent（内含 LLM 客户端 + 记忆库连接），避免每请求重建
_agent: TicketAgent | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """生命周期钩子：启动时装载配置与 Agent，退出时释放。

    为什么不用 @app.on_event("startup")：那是旧写法，FastAPI 已标记弃用。
    """
    global _agent
    load_dotenv()
    setup_logging("INFO")
    _agent = TicketAgent(Settings.from_env())
    yield
    _agent = None


app = FastAPI(
    title="电商售后工单 Agent",
    version="0.1.0",
    description="有界自主 Agent 的 HTTP 接口：含 RAG 政策检索、长期记忆、L2 人工确认。",
    lifespan=lifespan,
)


# ---------------- 请求 / 响应模型（Pydantic 自动校验 + 自动生成文档） ----------------


class ChatRequest(BaseModel):
    text: str = Field(..., max_length=2000, description="用户输入")
    user_id: str = Field(default="anonymous", max_length=64, description="用户标识，决定记忆归属")


class ConfirmRequest(BaseModel):
    user_id: str = Field(default="anonymous", max_length=64)
    # 客户端需把 /chat 返回的 pending 动作原样回传；服务端据此幂等执行
    pending: dict[str, Any] = Field(..., description="POST /chat 返回的 pending 动作原样回传")


# ---------------- 端点 ----------------


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查：容器编排 / 负载均衡靠它判断服务是否存活。"""
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """对话：一次请求 = 一次 Agent 调用。

    返回里若 pending 非空，说明有 L2 危险动作待人工确认，
    客户端应展示给用户，用户确认后再调 /confirm。
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    resp = _agent.handle(req.text, user_id=req.user_id)
    return {
        "behavior": resp.behavior,
        "output": resp.output,
        "intent": resp.intent,
        "tools_called": resp.tools_called,
        "steps": resp.steps,
        "stop_reason": resp.stop_reason,
        "pending": resp.pending,
        "trace_id": resp.trace_id,
        "latency_ms": resp.latency_ms,
        "cost": resp.cost,
        "degraded": resp.degraded,
    }


@app.post("/confirm")
def confirm(req: ConfirmRequest) -> dict[str, Any]:
    """执行 /chat 返回的待确认动作。

    幂等：同一条 pending（同一幂等键）重复提交不会重复建单。
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    try:
        outcome = _agent.confirm(req.pending, user_id=req.user_id)
    except Exception as exc:  # 兜底：任何异常都转成结构化错误，不让服务崩
        raise HTTPException(status_code=400, detail=f"确认执行失败：{exc}") from exc
    return {"ok": True, "result": outcome}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
