"""RAG 检索引擎：把 data/policies.md 变成可语义检索的知识库。

设计原则（和工具层一致）：
- 这个文件只做一件事：给定一句话 query，返回最相关的 top_k 条政策。
- 不关心"谁在查""权限是什么"——那是 tools.py / loop.py 的事。
- 对外只暴露一个函数：retrieve(query, top_k=3)。

内部四件套（思路和 tutorials/day12_embed.py 一样，但这里是生产版、可被 import）：
  - load_chunks  : 按 '## ' 标题把 policies.md 切成一条条政策
  - _chroma_recall: 用 Chroma 向量库做粗召回（首选路径）
  - _get_vectors : 把每条政策算成向量；算过就缓存到磁盘，语料不变不重算（降级路径）
  - cosine       : 算"两句意思靠多近"（余弦相似度，降级时用）

检索为什么改用 Chroma：
  原来是把向量全读进内存、手写余弦循环全量扫描。语料一大就慢，而且"自己算余弦"
  在简历上不算"用过向量数据库"。Chroma 是纯本地的（一个 pip 包，不必装服务器），
  内部用 HNSW 索引做近似最近邻，毫秒级返回。

  ⚠ 关键设计：embedding 仍然由我们自己的硅基流动 bge-m3 生成，再"喂"给 Chroma
  （add/query 都显式传 embeddings）。绝不能让 Chroma 用它的默认模型自己 embed ——
  那样会引入第二个 embedding 模型，向量空间不一致，检索结果直接失真。

  Chroma 只负责"存 + 查"，不负责"理解"。
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from openai import OpenAI

from .config import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
POLICIES_FILE = ROOT / "data" / "policies.md"
CACHE_FILE = ROOT / "data" / "policy_vectors.json"
BATCH_SIZE = 5
EMBED_TIMEOUT_S = 30.0
RERANK_TOP_K = 8            # 向量粗召回多少条，交给 rerank 精排（召回宁多勿漏）
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Chroma 向量库：持久化到本地目录，进程重启后仍在，无需任何服务器
CHROMA_DIR = ROOT / "data" / "chroma_db"
CHROMA_COLLECTION = "policies"


def load_chunks() -> list[dict[str, str]]:
    """按 '## ' 切块。每块 = 一条政策（id / 标题 / 正文）。"""
    text = POLICIES_FILE.read_text(encoding="utf-8")
    chunks: list[dict[str, str]] = []
    for block in text.split("\n## ")[1:]:
        head, _, body = block.partition("\n")
        parts = head.split(" ", 1)
        chunks.append(
            {
                "id": parts[0],
                "title": parts[1] if len(parts) > 1 else head,
                "text": head + "\n" + body.strip(),
            }
        )
    return chunks


def _embed(texts: list[str]) -> list[list[float]]:
    """调硅基流动的 bge-m3，把一批文本变成向量（一串数字）。"""
    load_dotenv()  # 把 .env 里的 EMBEDDING_* 塞进环境变量
    client = OpenAI(
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        base_url=os.environ.get("EMBEDDING_BASE_URL", ""),
        timeout=EMBED_TIMEOUT_S,
    )
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    """把一句话变成向量，供语义缓存 / 检索复用（禁止让调用方自己拼 batch）。"""
    return _embed([text])[0]


def _get_vectors() -> tuple[list[str], list[list[float]], list[dict[str, str]]]:
    """读缓存，或现算并写缓存。返回 (ids, vectors, chunks)。"""
    chunks = load_chunks()
    source_hash = hashlib.sha256(POLICIES_FILE.read_bytes()).hexdigest()
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if cache.get("source_hash") == source_hash:
                return cache["ids"], cache["vectors"], chunks
        except (json.JSONDecodeError, KeyError):
            pass  # 缓存坏了就当没有，重新算，别让它拖垮检索
    vectors = _embed([c["text"] for c in chunks])
    CACHE_FILE.write_text(
        json.dumps(
            {"source_hash": source_hash, "ids": [c["id"] for c in chunks], "vectors": vectors},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return [c["id"] for c in chunks], vectors, chunks


# Chroma 建库标记：记下"建库时语料的内容哈希"，用来判断语料是否被改动过
# （不依赖 collection.metadata —— 那个在各版本行为不一致，用文件最稳）
CHROMA_MARKER = CHROMA_DIR / "source_hash.txt"


def _chroma_recall(
    q_vec: list[float], chunks: list[dict[str, str]], source_hash: str, n: int
) -> list[int] | None:
    """用 Chroma 做粗召回，返回 chunks 下标（按相似度降序）。

    首次调用会把语料 embed 后写进 Chroma；之后直接查，不必重算全库向量。
    语料条数变化或内容哈希变化，自动重建。

    任何异常一律返回 None，让上层降级回「JSON 缓存 + 余弦」——
    和 _rerank 的降级思路一致：检索永远要有结果，不能因为向量库挂了就整体瘫痪。
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(CHROMA_COLLECTION)

    marker_ok = CHROMA_MARKER.exists() and CHROMA_MARKER.read_text(encoding="utf-8").strip() == source_hash
    if col.count() != len(chunks) or not marker_ok:
        # 重建：语料变了，必须重新 embed（这一步会调 embedding API，是唯一有成本的路径）
        vectors = _embed([c["text"] for c in chunks])
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=vectors,
        )
        CHROMA_MARKER.write_text(source_hash, encoding="utf-8")

    res = col.query(query_embeddings=[q_vec], n_results=min(n, len(chunks)))
    id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
    return [id_to_idx[cid] for cid in res["ids"][0] if cid in id_to_idx]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：两串数字越"同向"越接近 1，越"无关"越接近 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _rerank(query: str, docs: list[str]) -> list[tuple[int, float]] | None:
    """调硅基流动 bge-reranker-v2-m3 做精排。

    返回 (docs 内下标, 相关性分数) 按分数降序；任何失败返回 None，
    让上层退回纯向量排序——和 search_policy 的降级思路一致，保证检索永远有结果。
    """
    if not docs:
        return []
    load_dotenv()
    key = os.environ.get("EMBEDDING_API_KEY", "")
    base = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
    if not key or not base:
        return None
    body = json.dumps(
        {"model": RERANK_MODEL, "query": query, "documents": docs}
    ).encode()
    req = urllib.request.Request(
        base + "/rerank",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT_S) as r:
            data = json.load(r)
    except Exception:
        return None
    results = data.get("results", [])
    # 接口按 documents 下标顺序返回；我们按 relevance_score 重新降序排
    ranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    return [(x["index"], x["relevance_score"]) for x in ranked]


def retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """对外唯一接口：一句话 → 最相关的 top_k 条政策。

    两阶段：
      1) 向量粗召回 top(RERANK_TOP_K) 条（快，但顺序未必准）；
      2) rerank 精排这 RERANK_TOP_K 条，取真正最相关的 top_k。
    rerank 不可用时退回第一阶段结果（降级）。
    """
    chunks = load_chunks()
    source_hash = hashlib.sha256(POLICIES_FILE.read_bytes()).hexdigest()
    q_vec = _embed([query])[0]  # 把用户的问题也变成向量

    # 第一阶段：向量粗召回。优先 Chroma；它挂了就降级回「JSON 缓存 + 余弦」
    top_idx: list[int] | None = None
    try:
        top_idx = _chroma_recall(q_vec, chunks, source_hash, RERANK_TOP_K)
    except Exception:
        top_idx = None  # 不吞掉信息：下面会走改造前的老路径，功能不回退
    if not top_idx:
        _ids, vectors, chunks = _get_vectors()
        coarse = sorted(
            ((cosine(q_vec, v), i) for i, v in enumerate(vectors)),
            key=lambda x: x[0],
            reverse=True,
        )
        top_idx = [i for _, i in coarse[:RERANK_TOP_K]]

    candidates = [chunks[i] for i in top_idx]
    # 第二阶段：rerank 精排
    reranked = _rerank(query, [c["text"] for c in candidates])
    if reranked is None:
        final_idx = top_idx[:top_k]  # 降级：用向量分数
    else:
        final_idx = [top_idx[ri] for ri, _ in reranked[:top_k]]
    return [
        {"id": chunks[i]["id"], "title": chunks[i]["title"], "body": chunks[i]["text"]}
        for i in final_idx
    ]
