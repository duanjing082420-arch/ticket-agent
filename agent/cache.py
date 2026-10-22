"""语义缓存：用"意思相似"而非"字面相等"来复用历史答案。

为什么需要它（来自项目痛点）：每次提问都要跑 LLM（按 token 计费 + 1~5 秒延迟）。
客服场景大量问题语义重复（"怎么退货" / "申请退款" / "退钱流程"），把它们算成向量、
按余弦相似度找历史答案，命中就直接返回，连 LLM 都不调 —— 省钱、变快、还能扛 API 限流。

设计要点（和 RAG/Chroma 同一套哲学）：
- 缓存永远不能拖垮主链路：embedding 失败 / 后端不可用，一律降级成"未命中"，
  走原来的 LLM 路径，绝不抛错给编排层。
- 后端可插拔：Upstash（云 Redis，零本地安装，REST 调用）或 InMemory（零依赖，便于本地验证）。
- 用 TTL 过期，避免政策更新后还吐旧答案（脏数据）。

为什么不用普通字典精确匹配：字面不等的同义问法会扑空，缓存命中率极低。
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any

from .config import Settings
from .observability import emit


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：两串数字越"同向"越接近 1，越"无关"越接近 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class CacheBackend:
    """后端协议：put 存一条，scan 取全部（相似度由上层算）。"""

    def put(self, entry: dict[str, Any]) -> None:  # pragma: no cover - 接口占位
        raise NotImplementedError

    def scan(self) -> list[dict[str, Any]]:  # pragma: no cover - 接口占位
        raise NotImplementedError


class InMemoryBackend(CacheBackend):
    """零依赖的内存后端：进程存活期间有效，重启清空。用于本地验证 / 单测。"""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def put(self, entry: dict[str, Any]) -> None:
        key = hashlib.md5(entry["query"].encode("utf-8")).hexdigest()[:16]
        self._store[key] = entry

    def scan(self) -> list[dict[str, Any]]:
        return list(self._store.values())


class UpstashBackend(CacheBackend):
    """Upstash Redis（Serverless，REST 调用，零本地安装）。

    每条问答存为一个 key（带 EX 过期）；另用一个 index key 维护所有 key 的列表，
    查询时先取 index 再批量 GET。TTL 到期后 key 自动消失，scan 时遇到的空值直接跳过 ——
    这正好演示了"Redis 过期"的效果。
    """

    def __init__(self, url: str, token: str, ttl: int) -> None:
        self.url = url
        self.token = token
        self.ttl = ttl
        self._index_key = "qa:index"

    def _post(self, commands: list[list[Any]]) -> list[dict[str, Any]]:
        # Upstash REST 管道端点：POST {url}/pipeline，请求体是「裸二维 JSON 数组」。
        # 注意不是 {"commands": [...]} —— 那是错的（实测返回 HTTP 400）。
        # 响应是同顺序的数组：[{"result": ...}, ...]
        req = urllib.request.Request(
            self.url.rstrip("/") + "/pipeline",
            data=json.dumps(commands).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def put(self, entry: dict[str, Any]) -> None:
        key = "qa:" + hashlib.md5(entry["query"].encode("utf-8")).hexdigest()[:16]
        self._post([["SET", key, json.dumps(entry, ensure_ascii=False), "EX", self.ttl]])
        try:
            idx_raw = self._post([["GET", self._index_key]])[0].get("result")
            keys = json.loads(idx_raw) if idx_raw else []
        except Exception:
            keys = []
        if key not in keys:
            keys.append(key)
        # index 活得比单条长，避免短期集中过期后 index 还在但内容没了
        self._post([["SET", self._index_key, json.dumps(keys), "EX", max(self.ttl, 7 * 86400)]])

    def scan(self) -> list[dict[str, Any]]:
        try:
            idx_raw = self._post([["GET", self._index_key]])[0].get("result")
        except Exception:
            return []
        if not idx_raw:
            return []
        keys = json.loads(idx_raw)
        if not keys:
            return []
        try:
            res = self._post([["GET", k] for k in keys])
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for item in res:
            raw = item.get("result") if isinstance(item, dict) else None
            if not raw:
                continue  # 已过期 / 被清理的 key，跳过
            try:
                out.append(json.loads(raw))
            except Exception:
                pass
        return out


class SemanticCache:
    def __init__(self, settings: Settings) -> None:
        self.enabled = False
        self.threshold = settings.cache_sim_threshold
        self.backend: CacheBackend | None = None
        self._build_backend(settings)

    def _build_backend(self, s: Settings) -> None:
        if not s.cache_enabled:
            emit("cache_disabled", reason="env_off")
            return
        if s.cache_backend == "memory":
            self.backend = InMemoryBackend()
            self.enabled = True
            emit("cache_enabled", backend="memory")
            return
        if s.cache_backend == "upstash":
            if s.upstash_url and s.upstash_token:
                self.backend = UpstashBackend(s.upstash_url, s.upstash_token, s.cache_ttl_s)
                self.enabled = True
                emit("cache_enabled", backend="upstash")
            else:
                # 配了 upstash 却没给凭据：不要崩，降级关闭并告警
                emit("cache_disabled", reason="upstash_creds_missing")
            return
        emit("cache_disabled", reason="unknown_backend")

    def get(self, query: str) -> str | None:
        """语义查缓存：返回相似历史答案；未命中 / 出错一律返回 None（降级不破主链路）。"""
        if not self.enabled or self.backend is None:
            return None
        try:
            from .rag import embed_query

            q_vec = embed_query(query)
        except Exception as exc:
            emit("cache_embed_failed", error=type(exc).__name__)
            return None
        try:
            entries = self.backend.scan()
        except Exception as exc:
            emit("cache_scan_failed", error=type(exc).__name__)
            return None
        best: dict[str, Any] | None = None
        best_sim = -1.0
        for e in entries:
            vec = e.get("vec")
            if not vec:
                continue
            sim = _cosine(q_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best = e
        if best is not None and best_sim >= self.threshold:
            emit("cache_hit", sim=round(best_sim, 3))
            return best["answer"]
        emit("cache_miss", best_sim=round(best_sim, 3) if best is not None else None)
        return None

    def put(self, query: str, answer: str) -> None:
        if not self.enabled or self.backend is None:
            return
        try:
            from .rag import embed_query

            q_vec = embed_query(query)
        except Exception as exc:
            emit("cache_embed_failed", error=type(exc).__name__)
            return
        try:
            self.backend.put({"query": query, "vec": q_vec, "answer": answer})
            emit("cache_put")
        except Exception as exc:
            emit("cache_put_failed", error=type(exc).__name__)
