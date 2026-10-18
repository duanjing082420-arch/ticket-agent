"""
Day 9b：UUID（Idempotency-Key）到底该在什么时候生成？

不联网、不花钱、3 秒跑完。

先分清两件事：
    订单号  → 决定「退哪一笔」
    UUID    → 决定「这是同一次提交，还是新的一次提交」

本脚本模拟一个退款服务，跑三个场景看结果。
"""

import uuid


# ============================================================
# 一个极简的退款服务（用内存字典假装是数据库）
# ============================================================
class RefundService:
    def __init__(self, use_business_rule: bool = False):
        self.processed_keys = {}      # {UUID: 结果}  ← 幂等表
        self.pending_orders = set()   # 正在进行中的退款订单  ← 业务规则表
        self.use_business_rule = use_business_rule

    def refund(self, order_id: str, client_key: str) -> str:
        """返回这次到底有没有真的退钱"""

        # ---- 第 1 道防线：幂等表（Idempotency-Key）----
        if client_key in self.processed_keys:
            return f"skipped  这笔已经处理过（UUID 复用），未重复执行"

        # ---- 第 2 道防线：业务规则 ----
        # 一个订单同时只能有一笔「进行中」的退款
        if self.use_business_rule and order_id in self.pending_orders:
            return f"skipped  订单 {order_id} 已有一笔进行中的退款，拒绝重复申请"

        # ---- 真的执行 ----
        self.processed_keys[client_key] = "done"
        self.pending_orders.add(order_id)
        return f"done     已为 {order_id} 退款"


def show(title: str, service, events):
    """events 里每一项 = (订单号, 这次请求带的 UUID)"""
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print("=" * 64)
    real = 0
    for i, (order_id, key) in enumerate(events, 1):
        r = service.refund(order_id, key)
        if r.startswith("done"):
            real += 1
        print(f"  第 {i} 次  UUID={key[:8]}…  → {r}")
    print(f"  ── 真正退了 {real} 次钱")


print("\n问题：UUID 该在什么时候生成？\n")


# ============================================================
# 场景 1：网络重试 / 超时重发 —— 5 次请求复用同一个 UUID
# ============================================================
# 用户点了 1 次按钮 → 前端生成 1 个 UUID → 网络不好重发了 5 次
one_key = str(uuid.uuid4())
show("场景 1：用户点了 1 次，网络重试 5 次（复用同一个 UUID）",
     RefundService(),
     [("SO001", one_key)] * 5)


# ============================================================
# 场景 2：用户真的狂点 5 次 —— 每次都生成新 UUID
# ============================================================
# 这时候 Idempotency-Key 拦不住了！因为每次都是「新的一次提交」
show("场景 2：用户真狂点 5 次，每次前端都新生成 UUID",
     RefundService(),
     [("SO001", str(uuid.uuid4())) for _ in range(5)])


# ============================================================
# 场景 3：加上业务规则 —— 一个订单只能有一笔进行中退款
# ============================================================
show("场景 3：同样狂点 5 次，但服务端加了业务规则",
     RefundService(use_business_rule=True),
     [("SO001", str(uuid.uuid4())) for _ in range(5)])


# ============================================================
# 场景 4：用户真的想退第二笔（换了个订单）
# ============================================================
# 这是正常业务，必须放行 —— 不能因为「防重」把正常需求也挡了
show("场景 4：用户换了个订单，重新点一次（正常业务，必须放行）",
     RefundService(use_business_rule=True),
     [("SO001", str(uuid.uuid4())), ("SO002", str(uuid.uuid4()))])


print(f"\n{'=' * 64}")
print("  答案")
print("=" * 64)
print("""
  生成 UUID 的时机 = 「用户表达了新的一次提交意图」的时刻。

    用户点一次「申请退款」按钮  → 生成一次 UUID
    之后网络重试 / 超时重发    → 复用同一个 UUID（这才是它要挡的东西）
    用户重新打开页面再点一次   → 新 UUID

  所以你那两个问题的答案是相反的：

    「狂点 5 次」若是同一次点击被重发  → UUID 一样 → 挡得住 ✅（场景 1）
    「狂点 5 次」若是真的按了 5 下     → UUID 不一样 → 挡不住 ❌（场景 2）

  而场景 2 恰恰是最常见的情况。所以光有 Idempotency-Key 不够，
  必须再叠两层：

    前端：点了就置灰按钮 / 防抖（不让第 2 次请求发出去）
    服务端：业务状态机 —— 一个订单同时只能有一笔「进行中」的退款（场景 3）

  最后一条才是真正的兜底。它是业务规则，不是算法技巧：
  不管你 UUID 怎么生成、网络怎么重试，订单状态不允许，就是退不了。

  ⚠ 反面教训：别把防重做得太狠。场景 4 里用户换订单重新申请是
    正常需求，必须放行。「防重复」不等于「只准退一次」。
""")
