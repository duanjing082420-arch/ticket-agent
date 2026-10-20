"""
Day 12 第二步：BM25 打分排序 —— 把「召回 15 条」变成「排好序的 Top-3」。

不联网、不花钱、不装任何包（纯标准库）。
本脚本自带与第一步关键词匹配的对照，直接跑即可看到差距。

跑法：
    python tutorials/day12_bm25.py
"""

import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

POLICIES_FILE = Path(__file__).resolve().parent.parent / "data" / "policies.md"

# BM25 的两个经验参数，业界默认值，不要随意调
K1 = 1.5   # 词频饱和度：同一个词在文档里出现再多次，贡献也不会无限增加
B = 0.75   # 长度归一化：长文档不应该仅仅因为词多就得分高


def tokenize(text: str) -> list[str]:
    """
    中文用字符 bigram 切分，避开 jieba 依赖。

    为什么是 bigram 而不是单字：单字会把"退货"拆成"退"和"货"，
    大量无关文档只要含"退"或"货"就被命中——这正是第一步踩的坑。
    bigram 保留了相邻字的组合信息，"退款"不会被拆散。
    """
    segs = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)
    tokens = []
    for s in segs:
        if "\u4e00" <= s[0] <= "\u9fff":
            if len(s) == 1:
                tokens.append(s)
            else:
                tokens.extend(s[i : i + 2] for i in range(len(s) - 1))
        else:
            tokens.append(s.lower())
    return tokens


def load_chunks() -> list[dict[str, str]]:
    """按 '## ' 标题切块。每块 = 编号 + 标题 + 正文。"""
    text = POLICIES_FILE.read_text(encoding="utf-8")
    chunks = []
    for block in re.split(r"\n## ", text)[1:]:
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


class BM25:
    """标准 BM25，纯标准库实现。语料只有 15 条，全量扫描足够快。"""

    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avgdl = sum(self.lens) / self.n if self.n else 0.0
        self.df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        # 平滑版 IDF，避免 df=0 时除零
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for d, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in set(query_tokens):
                f = d.count(t)
                if f == 0:
                    continue
                idf = self._idf(t)
                s += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self.avgdl))
            out.append(s)
        return out


def legacy_hits(chunks: list[dict[str, str]], keyword: str) -> int:
    """第一步的旧逻辑，只用来打印对照数字。"""
    keyword = keyword.strip()
    hits = [c for c in chunks if keyword in c["text"]]
    if not hits:
        hits = [c for c in chunks if any(ch in c["text"] for ch in keyword)]
    return len(hits)


QUERIES = [
    "退款多久到账",
    "七天无理由退货怎么算",
    "我买的鞋子不合适能退吗",
    "快递一直不更新怎么办",
    "生鲜坏了赔不赔",
    "东西烂了找谁",
]

if __name__ == "__main__":
    chunks = load_chunks()
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25(corpus)

    print(f"语料：{len(chunks)} 条，切块方式：按 '## ' 标题\n")

    for q in QUERIES:
        q_tokens = tokenize(q)
        scores = bm25.scores(q_tokens)
        ranked = sorted(
            zip(scores, chunks), key=lambda x: x[0], reverse=True
        )
        top3 = ranked[:3]
        best = top3[0][0]

        print(f"问法：{q}")
        print(f"  旧逻辑召回 {legacy_hits(chunks, q)}/{len(chunks)} 条 → BM25 取 Top-3")
        for score, c in top3:
            if score <= 0:
                continue
            mark = "  ◀ 首选" if score == best else ""
            print(f"    {score:6.2f}  {c['id']} {c['title']}{mark}")
        if best <= 0:
            print("    （全部得分 0 —— BM25 一个都没找到）")
        print()

    print("=" * 60)
    print("该看的两件事：")
    print("  1. 前 5 个问法：召回从 15 条变成了有分数的 3 条，且最相关的排在首位")
    print("  2. 最后一个问法「东西烂了找谁」：BM25 得分 0")
    print("     因为用户说「烂了」，文档写的是「腐烂变质」")
    print("     字面上没有任何重叠，BM25 再怎么算也连不上")
    print("这就是字面匹配的死穴，也是下一步「查询扩展」要解决的目标")
