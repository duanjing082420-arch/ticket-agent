"""存储后端抽象：把"长期记忆存在哪"和"业务怎么用"彻底解耦。

为什么单独一层（来自 tianjin/app/storage/db.py 注释铺好的路）：
    项目早期用标准库 sqlite3 降级（零安装），但生产上 MySQL 才是常规。
    只要把"存储"收敛到这一层、对外只暴露 save_turn/load_history/clear 三个方法，
    以后换 TiDB/MySQL 只是多实现一个后端，service.py 一行都不用动。

本文件提供两个后端：
- SqliteMemoryBackend：本地单文件（零依赖，评测隔离库、默认走它）
- TidbHttpMemoryBackend：TiDB Cloud Serverless 的 SQL-over-HTTP 接口（/v1beta/sql），
  不用装任何 MySQL 驱动，urllib 直连，等价于"云上 MySQL"。

两个后端对外行为完全一致（同表结构、同方法签名），MemoryStore 按配置二选一。
"""

from __future__ import annotations

import abc
import base64
import json
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Iterable


# --------------------------------------------------------------------------
# 抽象接口：业务层只认这三个方法
# --------------------------------------------------------------------------

class MemoryBackend(abc.ABC):
    """长期记忆后端接口。所有实现必须保证 save/load/clear 语义一致。"""

    @abc.abstractmethod
    def save_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        ...

    @abc.abstractmethod
    def load_history(self, user_id: str, limit: int = 6) -> list[dict[str, str]]:
        ...

    @abc.abstractmethod
    def clear(self, user_id: str) -> int:
        ...


# --------------------------------------------------------------------------
# 后端一：SQLite（零依赖，默认）
# --------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "memory.db"


class SqliteMemoryBackend(MemoryBackend):
    """每个 user_id 一份对话流水账（episodic memory）。单文件、能直接 sqlite3 打开看。"""

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_user "
                "ON memory_turns(user_id, id)"
            )

    def save_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        now = int(time.time())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO memory_turns(user_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                (user_id, "user", user_text, now),
            )
            conn.execute(
                "INSERT INTO memory_turns(user_id, role, content, created_at) "
                "VALUES (?,?,?,?)",
                (user_id, "assistant", assistant_text, now),
            )

    def load_history(self, user_id: str, limit: int = 6) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM memory_turns
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit * 2),
            ).fetchall()
        rows = list(reversed(rows))
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def clear(self, user_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM memory_turns WHERE user_id = ?", (user_id,)
            )
            return cur.rowcount


# --------------------------------------------------------------------------
# 后端二：TiDB Cloud Serverless（MySQL 协议兼容，SQL-over-HTTP，零驱动）
# --------------------------------------------------------------------------

def _escape_sql_string(value: str) -> str:
    """MySQL 字符串字面量转义（防 SQL 注入）。

    我们不用驱动的参数化（/v1beta/sql 的 params 行为各版本不一致），
    改为对自由文本做标准 C API 转义后内联，覆盖反斜杠/引号/控制字符。
    连 user_id 也走这里（它来自请求体，不可信）。
    """
    if value is None:
        return "NULL"
    s = value
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\x00", "\\x00")
    s = s.replace("\x1a", "\\x1a")
    return "'" + s + "'"


class TidbHttpMemoryBackend(MemoryBackend):
    """用 TiDB Serverless 的 /v1beta/sql HTTP 接口当 MySQL 用，无需任何驱动。

    请求格式（已核对官方 serverless-js 驱动与 TiDB Cloud Zero 文档）：
        POST https://http-<host>/v1beta/sql
        Authorization: Basic base64(user:password)
        TiDB-Database: <database>
        Content-Type: application/json
        {"query": "SELECT ..."}

    鉴权/端点/表结构都失败即抛错——由上层 MemoryStore 捕获并降级回 SQLite，
    绝不因为"云端连不上"就让整个服务起不来。
    """

    def __init__(
        self,
        sql_url: str,
        user: str,
        password: str,
        database: str,
        timeout: float = 10.0,
    ) -> None:
        if not sql_url:
            raise ValueError("tidb_sql_url 为空，无法使用 TiDB 后端")
        self.sql_url = sql_url.rstrip("/")
        self._auth = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        self._database = database
        self._timeout = timeout
        self._ensure_schema()

    # —— 网络层 ——

    def _post(self, query: str) -> dict:
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            self.sql_url,
            data=body,
            headers={
                "Authorization": "Basic " + self._auth,
                "TiDB-Database": self._database,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                raw = r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"TiDB HTTP {e.code}: {detail}") from e
        except Exception as e:  # 网络抖动 / TLS 中断等
            raise RuntimeError(f"TiDB 网络错误: {type(e).__name__}: {e}") from e
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"TiDB 响应非 JSON: {raw[:200]}") from e
        self._raise_if_error(resp)
        return resp

    @staticmethod
    def _raise_if_error(resp: dict) -> None:
        # 错误可能以顶层 error 或 results[].error 形式返回
        if isinstance(resp.get("error"), str) and resp["error"]:
            raise RuntimeError(f"TiDB 错误: {resp['error']}")
        for r in resp.get("results", []) or []:
            if isinstance(r.get("error"), str) and r["error"]:
                raise RuntimeError(f"TiDB 语句错误: {r['error']}")

    @staticmethod
    def _extract_rows(resp: dict) -> list[dict]:
        """兼容多种返回形状，把结果统一成 [{列:值}, ...]。

        ⚠ 形状 C 是 2026-10-23 真实联调才钉死的（离线冒烟永远测不出来）：
        TiDB Serverless /v1beta/sql 返回的是**顶层** types[].name + rows[][]，
        既没有 results 也没有 data。早期版本只认 A/B 两种，最后掉进
        "rows 是列表就原样返回" 的兜底分支，吐出 [["..."]] 这种裸二维列表，
        业务层 r.get("role") 立刻 AttributeError。
        结论：第三方接口必须打一次真实请求，凭文档/记忆写解析器必踩坑。
        """
        # 形状 C：顶层 types + rows（TiDB Serverless /v1beta/sql 实测形态）
        types = resp.get("types")
        if isinstance(types, list) and types and isinstance(resp.get("rows"), list):
            names = [t.get("name") if isinstance(t, dict) else t for t in types]
            rows = resp["rows"]
            if rows and isinstance(rows[0], (list, tuple)):
                return [dict(zip(names, row)) for row in rows]
            if rows and isinstance(rows[0], dict):
                return rows
            return []
        # 形状 A：results[].fields + results[].rows（serverless-js 常见）
        if "results" in resp and resp["results"]:
            r0 = resp["results"][0]
            fields = r0.get("fields") or r0.get("columns") or []
            names = [f.get("name") if isinstance(f, dict) else f for f in fields]
            rows = r0.get("rows") or []
            if names and rows and isinstance(rows[0], (list, tuple)):
                return [dict(zip(names, row)) for row in rows]
            if rows and isinstance(rows[0], dict):
                return rows
        # 形状 B：data.columns + data.rows（Data Service 风格）
        if "data" in resp and isinstance(resp["data"], dict):
            data = resp["data"]
            cols = [c.get("col") if isinstance(c, dict) else c for c in data.get("columns", [])]
            rows = data.get("rows", [])
            if cols and rows and isinstance(rows[0], (list, tuple)):
                return [dict(zip(cols, row)) for row in rows]
            if rows and isinstance(rows[0], dict):
                return rows
        # 兜底：只接受「列表里是字典」的形态。
        # 绝不原样返回裸二维列表 —— 那会让调用方 r.get("role") 崩在半路。
        rows = resp.get("rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows
        return []

    def _ensure_schema(self) -> None:
        self._post(
            "CREATE TABLE IF NOT EXISTS memory_turns ("
            "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
            "user_id VARCHAR(128) NOT NULL, "
            "role VARCHAR(16) NOT NULL, "
            "content MEDIUMTEXT NOT NULL, "
            "created_at BIGINT NOT NULL, "
            "KEY idx_memory_user (user_id, id))"
        )

    # —— 接口实现 ——

    def save_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        now = int(time.time())
        u = _escape_sql_string(user_id)
        self._post(
            "INSERT INTO memory_turns(user_id, role, content, created_at) VALUES "
            f"({u}, 'user', {_escape_sql_string(user_text)}, {now}), "
            f"({u}, 'assistant', {_escape_sql_string(assistant_text)}, {now})"
        )

    def load_history(self, user_id: str, limit: int = 6) -> list[dict[str, str]]:
        rows = self._extract_rows(
            self._post(
                "SELECT role, content FROM memory_turns WHERE user_id = "
                f"{_escape_sql_string(user_id)} ORDER BY id DESC LIMIT {int(limit) * 2}"
            )
        )
        rows = list(reversed(rows))
        return [{"role": r.get("role", ""), "content": r.get("content", "")} for r in rows]

    def clear(self, user_id: str) -> int:
        resp = self._post(
            "DELETE FROM memory_turns WHERE user_id = "
            f"{_escape_sql_string(user_id)}"
        )
        # 受影响行数：2026-10-23 实测 DELETE 返回的是**顶层** rowsAffected
        # （{"rowsAffected":2,"lastInsertID":0,...}），不是文档里的 results[].affectedRows。
        # 早期版本只找后者，导致 clear() 恒返回 0 —— 真实响应优先于文档。
        for key in ("rowsAffected", "affectedRows", "affected_rows"):
            v = resp.get(key)
            if isinstance(v, int):
                return v
        for r in resp.get("results", []) or []:
            if isinstance(r.get("affectedRows"), int):
                return r["affectedRows"]
        return 0

    def health_check(self) -> None:
        """连通性探针：能跑 SELECT 1 即视为可用。"""
        self._post("SELECT 1")
