"""
Day 12 第一步：先看清关键词匹配的失败现场，再谈 RAG。

不联网、不花钱、不调 LLM。只做一件事：
用 tools.py 现有的子串匹配逻辑，在真实政策语料上跑几个真实问法，
看看它到底召回了几条。

跑法：
    python tutorials/day12_keyword_fail.py
"""

import re
import sys
from pathlib import Path

# Windows 下 print 到管道会退回 GBK，中文直接崩。先强制 UTF-8。
# 这是本项目 observability.force_utf8_stdout 处理过的同一个坑。
sys.stdout.reconfigure(encoding="utf-8")

POLICIES_FILE = Path(__file__).resolve().parent.parent / "data" / "policies.md"


def load_policies() -> list[dict[str, str]]:
    """把 policies.md 按 '## ' 标题切成条目。"""
    text = POLICIES_FILE.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n## ", text)[1:]:
        head, _, body = block.partition("\n")
        parts = head.split(" ", 1)
        out.append(
            {
                "id": parts[0],
                "title": parts[1] if len(parts) > 1 else head,
                "body": body.strip(),
            }
        )
    return out


def legacy_match(policies: list[dict[str, str]], keyword: str) -> list[dict[str, str]]:
    """
    原样复刻 agent/tools.py::_search_policy 的两段逻辑：

    第一段：keyword 作为整体子串去匹配标题或正文
    第二段（没命中时的退化）：把 keyword 拆成单个字符，任一字符出现在正文即算命中

    注意第二段的触发条件是"第一段没命中"，所以只要第一段命中了，
    第二段就不会执行——这也是为什么它的危害不容易被一眼看出来。
    """
    keyword = keyword.strip()
    hits = [p for p in policies if keyword in p["title"] or keyword in p["body"]]
    if not hits:
        hits = [p for p in policies if any(ch in p["body"] for ch in keyword)]
    return hits


QUERIES = [
    "退款多久到账",
    "七天无理由退货怎么算",
    "我买的鞋子不合适能退吗",
    "快递一直不更新怎么办",
    "生鲜坏了赔不赔",
]

if __name__ == "__main__":
    policies = load_policies()
    print(f"语料：{len(policies)} 条政策\n")

    for q in QUERIES:
        hits = legacy_match(policies, q)
        print(f"问法：{q}")
        print(f"  召回 {len(hits)}/{len(policies)} 条")
        for h in hits:
            print(f"    - {h['id']} {h['title']}")
        print()

    print("=" * 56)
    print("看输出时只需判断一件事：召回的这些里面，有几条是真正答得上这个问法的？")
    print("召回得多不等于检索得好——召回 10 条里只有 1 条相关，等于没检索，")
    print("只是把 15 条全塞给了模型，还白白多烧了 token。")
