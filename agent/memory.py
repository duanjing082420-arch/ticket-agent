"""记忆层：把每个用户的对话历史持久化到 SQLite，实现跨会话记忆。

为什么需要它：
现在的 TicketAgent.handle() 每次都是「失忆」的——用户上一句说「我是会员」，
下一句问「我的会员券能叠加吗」，它根本不记得。客服场景里用户经常跨多条消息
逐步补充信息（先说订单号，再说问题），没有记忆，体验就很碎。

两类记忆（面试常考）：
- 短期记忆（working memory）：同一次 handle() 内部，loop.py 已经在 messages 里
  保留了工具往返——那本身就是短期记忆。
- 长期记忆（long-term memory）：跨多次 handle() / 跨会话，需要落盘。
  本项目用 SQLite 单文件存，轻量、零依赖、能直接 `sqlite3` 打开看。

设计取舍（第一步保持简单）：
- 只存「用户说了什么 / 助手答了什么」这一层（episodic memory），
  不存工具内部消息（那些是短期、一次性的，不该污染长期记忆）。
- 不做「事实抽取」（semantic memory，需再调一次 LLM 提炼关键点），
  后面可以加一张 facts 表扩展。
- 读取只取最近 N 条，控制 token 成本，也避免把一年前的对话塞进 prompt。
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable

# DB 默认放在项目根 data/ 下，单文件，跟着项目走
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "memory.db"


class MemoryStore:
    """每个 user_id 一份对话流水账。

    线程安全：SQLite 连接级锁足够；本项目单进程调用，无需额外加锁。
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # check_same_thread=False：评测脚本可能跨线程跑；本项目实际单线程，影响可忽略
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
                    role       TEXT NOT NULL,            -- 'user' 或 'assistant'
                    content    TEXT NOT NULL,
                    created_at INTEGER NOT NULL         -- Unix 秒，便于排序/清理
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_user "
                "ON memory_turns(user_id, id)"
            )

    # ---------------- 写 ----------------

    def save_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """存一轮对话：先存用户说的，再存助手答的，保证时间顺序正确。"""
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

    # ---------------- 读 ----------------

    def load_history(self, user_id: str, limit: int = 6) -> list[dict[str, str]]:
        """取该用户最近的 limit 轮（一轮 = user + assistant），按时间正序返回。

        返回形如 [{'role':'user','content':...}, {'role':'assistant','content':...}, ...]
        可直接 extend 进 OpenAI messages。
        """
        with self._conn() as conn:
            # 一轮含两条，取最近 limit*2 条；先用 DESC 取最新，再反转回正序
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

    # ---------------- 维护 ----------------

    def clear(self, user_id: str) -> int:
        """清空某用户的历史（调试 / 隐私删除用）。返回删除条数。"""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM memory_turns WHERE user_id = ?", (user_id,)
            )
            return cur.rowcount
