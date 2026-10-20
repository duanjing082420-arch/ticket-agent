"""Day 13：单独测试 RAG 检索引擎（agent/rag.py）。

不启动整个 Agent，先确认"检索"这块本身是对的。
跑法：
    .venv/Scripts/python.exe tutorials/day13_rag_test.py
"""

import sys

sys.path.insert(0, ".")

from agent.rag import retrieve

QUERIES = [
    "东西烂了找谁",
    "退款多久到账",
    "鞋子不合适能退吗",
    "快递一直不更新怎么办",
]


def main() -> None:
    for q in QUERIES:
        hits = retrieve(q, top_k=3)
        print(f"\n问法：{q}")
        for h in hits:
            print(f"    {h['id']}  {h['title']}")


if __name__ == "__main__":
    main()
