"""day14：验证 rerank 精排是否比纯向量排序更准。

跑法：
    .venv/Scripts/python.exe tutorials/day14_rerank_test.py

这个脚本做两件事：
  1. 用"纯向量"取每个问法的前 3 条（rag.retrieve 加 rerank 之前就是这个结果）
  2. 用"向量 + rerank"取前 3 条（现在 rag.retrieve 的默认行为）
对比两者，看 rerank 把哪些题的顺序调对了。
"""

import sys

sys.path.insert(0, ".")

from agent.config import load_dotenv

load_dotenv()

import agent.rag as rag


def vector_only_top3(query: str) -> list[str]:
    """复刻 rerank 加进来之前的纯向量检索，作为对照基线。"""
    ids, vectors, chunks = rag._get_vectors()
    q_vec = rag._embed([query])[0]
    coarse = sorted(
        ((rag.cosine(q_vec, v), c) for v, c in zip(vectors, chunks)),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c["id"] for _, c in coarse[:3]]


def main() -> None:
    queries = [
        "鞋子不合适能退吗",
        "东西烂了找谁",
        "退款多久到账",
        "七天无理由退货从哪天算时间",
        "商品坏了能换吗",
    ]
    print("=" * 64)
    print("对比：纯向量 top3  vs  向量+rerank top3")
    print("=" * 64)
    for q in queries:
        vec = vector_only_top3(q)
        rk = rag.retrieve(q, top_k=3)
        rk_ids = [c["id"] for c in rk]
        changed = "  <<< rerank 改变了顺序" if vec != rk_ids else ""
        print(f"\n问法：{q}")
        print(f"  纯向量  : {vec}")
        print(f"  +rerank : {rk_ids}{changed}")
    print("\n" + "=" * 64)
    print("重点看「鞋子不合适能退吗」：rerank 应把 P-RETURN 顶到第一。")


if __name__ == "__main__":
    main()
