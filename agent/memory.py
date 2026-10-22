"""记忆层（编排入口）：对外仍是 MemoryStore，内部按配置选存储后端。

设计目标（来自 tianjin/app/storage/db.py 注释铺好的路）：
    业务层（service.py）只认 MemoryStore 的三个方法（save_turn/load_history/clear），
    至于"存在 SQLite 还是 TiDB/MySQL"由本文件在构造时决定——
    - 默认 sqlite：零依赖、评测隔离库、本地直接看；
    - 配置 memory_backend=mysql：走 TiDB Cloud Serverless 的 /v1beta/sql（云上 MySQL）；
    - 若 mysql 配置缺失或云端连不上，自动降级回 sqlite 并打点告警，服务不崩。

向后兼容：构造参数既可以是 Settings（生产/评测路径），也可以是 sqlite 文件路径
（tutorials/day15_memory_test.py 的用法），保证旧教程不崩。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .observability import emit
from .storage import (
    MemoryBackend,
    SqliteMemoryBackend,
    TidbHttpMemoryBackend,
)


class MemoryStore:
    """长期记忆分发器：把请求委派给具体的存储后端。"""

    def __init__(self, arg: Settings | str | Path | None = None) -> None:
        if isinstance(arg, Settings):
            self._init_from_settings(arg)
        else:
            # 兼容旧用法：传入 sqlite 文件路径（tutorials）或 None（用默认路径）
            db_path = arg if isinstance(arg, (str, Path)) else None
            self._backend: MemoryBackend = SqliteMemoryBackend(db_path)
            self.backend_name = "sqlite"

    def _init_from_settings(self, s: Settings) -> None:
        mysql_ok = (
            s.memory_backend == "mysql"
            and s.tidb_sql_url
            and s.tidb_user
            and s.tidb_password
            and s.tidb_database
        )
        if mysql_ok:
            try:
                be = TidbHttpMemoryBackend(
                    s.tidb_sql_url, s.tidb_user, s.tidb_password, s.tidb_database
                )
                be.health_check()  # 连通性探针：连不上立刻抛，进入降级
                self._backend = be
                self.backend_name = "tidb"
                emit("memory_backend_selected", backend="tidb")
                return
            except Exception as exc:  # 云端不可达 / 凭证错误 / 建表失败
                emit("memory_backend_fallback", to="sqlite", error=type(exc).__name__)
        self._backend = SqliteMemoryBackend(s.memory_db_path)
        self.backend_name = "sqlite"

    # ---------------- 写 ----------------

    def save_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        self._backend.save_turn(user_id, user_text, assistant_text)

    # ---------------- 读 ----------------

    def load_history(self, user_id: str, limit: int = 6) -> list[dict[str, str]]:
        return self._backend.load_history(user_id, limit)

    # ---------------- 维护 ----------------

    def clear(self, user_id: str) -> int:
        return self._backend.clear(user_id)
