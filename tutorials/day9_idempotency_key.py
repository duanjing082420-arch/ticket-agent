"""
Day 9：幂等键到底该怎么设计？

不联网、不花钱、3 秒跑完。

问题背景：
    day8 里我们的幂等键 = md5("refund:" + 订单号 + 退款理由)
    模型两次写的理由不一样（"不想要了" vs "用户不想要了"），键就不同 → 幂等失效 → 退两次钱

本脚本把 4 种设计方案放在同一批测试用例上跑，看谁能挡住、谁漏了。
"""

import hashlib
from enum import Enum


# ============================================================
# 测试用例：哪些应该被判定为「同一笔操作」？
# ============================================================
# 前 4 条：真实上是同一笔退款（用户狂点按钮 / 网络重试 / 换个说法再问一次）
# 后 2 条：真实上是不同的两笔（不同订单、或不同退款类型）
CASES = [
    # (订单号, 模型这次写出来的理由, 正确答案：和第一条是不是同一笔？)
    ("SO001", "不想要了", "—"),        # 基准
    ("SO001", "不想要了", True),       # 完全一样
    ("SO001", "用户不想要了", True),   # 换个说法
    ("SO001", "不想要了！", True),     # 多了一个标点
    ("SO001", "质量问题", False),      # 真的是另一笔（改成质量原因）
    ("SO002", "不想要了", False),      # 另一笔订单
]


# ============================================================
# 方案 A：现在 day8 的做法 —— 订单号 + 理由原文
# ============================================================
def key_a(order_id: str, reason: str) -> str:
    """问题：理由一个字不一样，键就变了 → 退两次钱"""
    raw = f"refund:{order_id}:{reason.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# ============================================================
# 方案 B：你提的做法 —— 关键词归一化
# ============================================================
# 把理由里的"关键词"抽出来当身份。识路对，但要维护这张表。
KEYWORDS = ["不想要", "质量问题", "发错货", "少件"]


def normalize_by_keyword(reason: str) -> str:
    """从理由里找关键词，找到就用关键词，找不到就退回原文"""
    for kw in KEYWORDS:
        if kw in reason:
            return kw
    return reason.strip()


def key_b(order_id: str, reason: str) -> str:
    raw = f"refund:{order_id}:{normalize_by_keyword(reason)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# ============================================================
# 方案 C：别让模型写自由文本 —— 改成枚举（enum）
# ============================================================
# 这是 Day 3 讲过的 schema 约束：能用 enum 就别用 string。
# 理由只能是这几个值之一，模型再怎么发挥也跑不出去。
class RefundReason(str, Enum):
    NOT_WANTED = "NOT_WANTED"   # 不想要了
    QUALITY = "QUALITY"         # 质量问题
    WRONG_ITEM = "WRONG_ITEM"   # 发错货
    MISSING = "MISSING"         # 少件


def key_c(order_id: str, reason_code: str) -> str:
    raw = f"refund:{order_id}:{reason_code}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# 把模型写的中文理由映射到枚举（真实项目里这一步也可以让模型直接输出 code）
def to_reason_code(reason: str) -> str:
    if "质量" in reason:
        return RefundReason.QUALITY.value
    if "发错" in reason:
        return RefundReason.WRONG_ITEM.value
    if "少" in reason:
        return RefundReason.MISSING.value
    return RefundReason.NOT_WANTED.value


# ============================================================
# 方案 D：正解 —— 幂等键由「发起方」生成，和内容无关
# ============================================================
# 用户点一次「申请退款」按钮 → 前端生成一次 UUID → 之后重试都复用同一个
# 服务端只认这个 key，完全不看内容。
# 业界标准叫 Idempotency-Key 请求头（Stripe / 微信支付 / 支付宝都是这么干的）
def key_d(client_key: str) -> str:
    # 注意：这里连订单号都不参与。因为"这一笔请求"的身份，是发起方给的，不是算出来的。
    return hashlib.md5(f"refund:{client_key}".encode("utf-8")).hexdigest()[:12]


# ============================================================
# 跑测试
# ============================================================
def run(name: str, keyfunc, args_list):
    """args_list 里每一项是传给 keyfunc 的参数元组"""
    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print("=" * 62)

    keys = [keyfunc(*a) for a in args_list]
    base = keys[0]

    ok = 0
    total = 0
    for i, (case, k) in enumerate(zip(CASES, keys)):
        order_id, reason, expected = case
        if expected == "—":
            print(f"  [基准] {order_id} / {reason:<14} → {k}")
            continue

        same = (k == base)
        total += 1
        right = (same == expected)
        if right:
            ok += 1
        mark = "✓" if right else "✗ 漏了"
        print(f"  {mark:<6} {order_id} / {reason:<14} → {k}  判定={same} 期望={expected}")

    print(f"  ── 结果：{ok}/{total} 正确")
    return ok, total


print("\n问题：用户狂点按钮 / 网络重试 / 换个说法再问一次，")
print("      服务端怎么认出「这是同一笔退款」？\n")

# 方案 A / B / C：都能从 CASES 直接取（订单号, 理由）
run("方案 A（day8 现状）：订单号 + 理由原文", key_a,
    [(c[0], c[1]) for c in CASES])

run("方案 B（你的提议）：订单号 + 关键词归一化", key_b,
    [(c[0], c[1]) for c in CASES])

run("方案 C：订单号 + 理由枚举 code（不让模型写自由文本）", key_c,
    [(c[0], to_reason_code(c[1])) for c in CASES])

# 方案 D：模拟「客户端生成的幂等键」
# 前 4 条是用户点一次按钮后反复重试 → 复用同一个 client_key
# 第 5、6 条是用户重新发起 → 前端生成新的 client_key
client_keys = [
    "c-7f3a-1111",   # 用户点了按钮，生成 UUID
    "c-7f3a-1111",   # 网络重试，复用同一个（← 关键）
    "c-7f3a-1111",   # 换个说法再问，还是同一次点击
    "c-7f3a-1111",   # 加个标点
    "c-9b2d-2222",   # 用户重新发起一笔（改成质量原因）→ 新 UUID
    "c-4e8c-3333",   # 另一笔订单 → 新 UUID
]
run("方案 D（正解）：幂等键由发起方生成，和内容无关", key_d,
    [(k,) for k in client_keys])


# ============================================================
# 补测：方案 B 的盲区
# ============================================================
# 上面那批用例里 B 拿了 5/5，但那是因为用例是用它的关键词表出的题。
# 换一批"同义但不含关键词"的说法，看它还行不行。
print(f"\n{'=' * 62}")
print("  补测：方案 B 的盲区（换个说法，但不含关键词表里任何一个词）")
print("=" * 62)

BASE_TEXT = "不想要了"
B_VARIANTS = [
    "不想买了",        # 人一看就知道是同一个意思
    "想退掉",
    "买错了",
    "多拍了一件",
    "不想要这个了",     # 这个含"不想要"，B 能挡住
]

base_key = key_b("SO001", BASE_TEXT)
print(f"  [基准] {BASE_TEXT:<14} → {base_key}")
for v in B_VARIANTS:
    k = key_b("SO001", v)
    same = (k == base_key)
    print(f"  {'✓ 挡住' if same else '✗ 漏了'}  {v:<14} → {k}")

print("""
  ↑ 5 个说法里有 4 个漏了。而这些话，任何一个人都看得出是「不想要了」。
    要挡住它们，你得把这张表加到几十上百条 —— 而且永远加不完。
    更糟的是误伤：真实的第二笔申请被判成重复 → 钱没退成，用户投诉。""")


print(f"\n{'=' * 62}")
print("  结论")
print("=" * 62)
print("""
  A 漏：理由文字一变就失效 —— 你发现的问题，确实存在
  B 漏：能挡一部分，但关键词表列不全，而且可能误伤（把两笔真
        实的不同请求判成同一笔 → 少退一次钱）
  C 稳：把自由文本换成枚举，模型跑不出集合外。但只能挡住"理由"，
        挡不住"用户真的想发起第二笔"
  D 正解：幂等键是「谁发起的这次请求」的身份证，由发起方生成一次、
        重试时复用。服务端不猜，只查表。

  真实项目是 C + D 一起用：
    · 用 enum 约束模型输出（C，防止自由文本污染业务字段）
    · 用客户端 Idempotency-Key 做幂等（D，真正的防重）
    · 再加一层业务兜底：一个订单同时只能有一笔「进行中」的退款
      （这条比任何哈希都可靠 —— 它是业务规则，不是算法技巧）
""")
