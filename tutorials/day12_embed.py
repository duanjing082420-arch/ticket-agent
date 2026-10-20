"""
Day 12 第三步：向量检索 —— 解决 BM25 连不上的「字面不像」问题。

前面两步的遗留：
  BM25 对「东西烂了找谁」得分全 0，因为用户说「烂了」，文档写「腐烂变质」，
  字面零重叠。本脚本用语义向量把这两个词连起来。

先跑 dry-run 验证链路（不联网、不花钱）：
    python tutorials/day12_embed.py --dry-run

拿到 key 后真跑：
    python tutorials/day12_embed.py

依赖：openai>=1.40.0,<2.0.0（项目已有）
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
POLICIES_FILE = ROOT / "data" / "policies.md"
CACHE_FILE = ROOT / "data" / "policy_vectors.json"

# 批次上限各家不同（阿里 v4 为 10、智谱为 64），保守取 5 保证跨平台可用
BATCH_SIZE = 5
EMBED_TIMEOUT_S = 30.0
EMBED_MAX_RETRIES = 2


def load_env_file(path: Path) -> dict[str, str]:
    """极简 .env 解析，让本脚本不必依赖 python-dotenv。"""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env


def load_chunks() -> list[dict[str, str]]:
    """按 '## ' 标题切块，与前两步完全一致。"""
    text = POLICIES_FILE.read_text(encoding="utf-8")
    chunks = []
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


def fake_vector(text: str, dim: int = 64) -> list[float]:
    """
    dry-run 专用：用文本做种子的确定性伪向量。

    它只有「同样输入得到同样输出」这一个性质，没有任何语义，
    所以 dry-run 下的检索结果是胡说的。它唯一的作用是验证链路通不通。
    """
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(dim)]


def embed_all(chunks: list[dict[str, str]], cfg: dict[str, str], dry_run: bool) -> list[list[float]]:
    """把全部语料向量化，结果落盘缓存。语料没变就不重算。"""
    source_hash = hashlib.sha256(POLICIES_FILE.read_bytes()).hexdigest()

    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if cache.get("source_hash") == source_hash and cache.get("model") == cfg["model"]:
                print(f"缓存命中：{CACHE_FILE.name}（语料与模型均未变，跳过向量化）")
                return cache["vectors"]
        except (json.JSONDecodeError, KeyError):
            # 缓存损坏就当没有，重新算一遍，不要让它把整个检索拖死
            print("缓存文件损坏，重新向量化")

    if dry_run:
        vectors = [fake_vector(c["text"]) for c in chunks]
        print(f"dry-run：生成 {len(vectors)} 条伪向量（维度 {len(vectors[0])}）")
        return vectors

    from openai import OpenAI

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=EMBED_TIMEOUT_S,
        max_retries=EMBED_MAX_RETRIES,
    )

    vectors: list[list[float]] = []
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            resp = client.embeddings.create(
                model=cfg["model"], input=[c["text"] for c in batch]
            )
            vectors.extend(item.embedding for item in resp.data)
            print(f"  已向量化 {len(vectors)}/{total}")
        except Exception as e:
            # 外部调用失败必须降级，不能让检索整体崩掉
            print(f"向量化失败：{type(e).__name__}: {e}")
            print("降级路径：改用 tutorials/day12_bm25.py 的 BM25 检索（不需要 embedding）")
            raise SystemExit(1) from e

    CACHE_FILE.write_text(
        json.dumps(
            {
                "model": cfg["model"],
                "source_hash": source_hash,
                "ids": [c["id"] for c in chunks],
                "vectors": vectors,
            }
        ),
        encoding="utf-8",
    )
    print(f"已缓存向量到 {CACHE_FILE.name}（下次语料不变直接读缓存）")
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    """纯标准库余弦相似度。15 条语料全量扫描足够快，不需要向量库。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


QUERIES = [
    "退款多久到账",
    "七天无理由退货怎么算",
    "我买的鞋子不合适能退吗",
    "快递一直不更新怎么办",
    "生鲜坏了赔不赔",
    "东西烂了找谁",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="不联网，用伪向量验证链路")
    args = parser.parse_args()

    env = {**load_env_file(ROOT / ".env"), **os.environ}
    cfg = {
        "base_url": env.get("EMBEDDING_BASE_URL", ""),
        "api_key": env.get("EMBEDDING_API_KEY", ""),
        "model": env.get("EMBEDDING_MODEL", ""),
    }

    chunks = load_chunks()
    print(f"语料：{len(chunks)} 条\n")

    if not args.dry_run:
        missing = [k for k, v in cfg.items() if not v]
        if missing:
            print(f"缺少配置：{', '.join(missing)}")
            print("请在 .env 里补上 EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL")
            print("或先跑：python tutorials/day12_embed.py --dry-run")
            raise SystemExit(1)

    vectors = embed_all(chunks, cfg, args.dry_run)

    if args.dry_run:
        # 伪向量无语义，query 也必须是伪向量，结果仅供验证链路
        q_vecs = [fake_vector(q) for q in QUERIES]
    else:
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            timeout=EMBED_TIMEOUT_S,
            max_retries=EMBED_MAX_RETRIES,
        )
        try:
            resp = client.embeddings.create(model=cfg["model"], input=QUERIES)
            q_vecs = [item.embedding for item in resp.data]
        except Exception as e:
            print(f"查询向量化失败：{type(e).__name__}: {e}")
            raise SystemExit(1) from e

    print()
    for q, qv in zip(QUERIES, q_vecs):
        scored = sorted(
            ((cosine(qv, v), c) for v, c in zip(vectors, chunks)),
            key=lambda x: x[0],
            reverse=True,
        )
        print(f"问法：{q}")
        for score, c in scored[:3]:
            print(f"    {score:6.4f}  {c['id']} {c['title']}")
        print()

    if args.dry_run:
        print("=" * 60)
        print("以上数字是伪向量算出来的，没有意义。dry-run 只证明链路通了：")
        print("  切块 → 向量化 → 缓存 → 余弦检索 → Top-3 输出")
        print("拿到 key 填进 .env 后，去掉 --dry-run 再跑，才是真实语义结果。")
        print("重点看「东西烂了找谁」这一条——BM25 那里是 0 分，这里应该能找到 P-FRESH。")


if __name__ == "__main__":
    main()
