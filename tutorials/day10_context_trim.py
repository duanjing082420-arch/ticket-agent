"""
Day 10：上下文（messages）太长了怎么办？

不联网、不花钱、3 秒跑完。

用户提议：「砍在出结果之后，压缩 AI 的回答，连着用户的问题一起发」
  ✅ 时机对：确实要在「发给模型之前」砍
  ✅ 保留用户问题对
  ⚠ 但要调三处：① 优先砍的是工具结果，不是 AI 回答
                ② 时机是「每一圈循环开头都检查」，不是等结果出来才砍
                ③ 有个真实的坑：砍了工具调用却留下工具结果 → 接口直接报错

本脚本演示四种砍法各自砍掉多少、会不会把消息砍坏。
"""

import copy

# 用字符数近似表示"长度"（真实项目要用 tiktoken 数 token，中文 1 字 ≈ 1 token）
def size(msg) -> int:
    c = msg.get("content") or ""
    if msg.get("tool_calls"):
        c += str(msg["tool_calls"])
    return len(c)


def total(msgs) -> int:
    return sum(size(m) for m in msgs)


# ============================================================
# 构造一段聊天记录：system + 3 轮对话 + 中间夹了工具调用
# ============================================================
def build_messages():
    return [
        {"role": "system", "content": "你是客服助手。只报告工具真实返回的数据，不要编造。"},
        {"role": "user", "content": "我上周买的耳机到哪了？"},
        {"role": "assistant", "content": "请提供订单号。"},
        {"role": "user", "content": "SO20240917001"},
        # ↓ 这一对必须成对存在：assistant 说"我要调工具" + tool 回"结果"
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "function": {"name": "get_order_status",
                                                     "arguments": '{"order_id":"SO20240917001"}'}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": '{"found": true, "order_id": "SO20240917001", "status": "已发货", '
                    '"item": "无线降噪耳机 Pro", "carrier": "顺丰速运", '
                    '"tracking_no": "SF1234567890", "shipped_at": "2024-09-18 09:05"}'},
        {"role": "assistant", "content": "已发货，顺丰 SF1234567890，可以凭单号查询物流轨迹。"},
        {"role": "user", "content": "那我能退货吗？"},
        {"role": "assistant", "content": "7 天无理由退货，需要看签收时间，我这边查不到签收时间。"},
        {"role": "user", "content": "帮我申请退款，理由是不想要了"},
    ]


def check_orphan(msgs):
    """检查有没有「孤儿」：有 tool 结果却没有对应的工具调用声明"""
    declared = set()
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            declared.add(tc["id"])
    orphan = [m for m in msgs if m.get("role") == "tool"
              and m.get("tool_call_id") not in declared]
    return orphan


def report(name, msgs, note=""):
    bad = check_orphan(msgs)
    status = f"❌ 坏了：{len(bad)} 条孤儿消息" if bad else "✅ 结构完好"
    print(f"\n  【{name}】  长度 {total(msgs)} 字  {status}")
    if note:
        print(f"     {note}")
    return len(bad)


# ============================================================
print("=" * 66)
print("  原始 messages")
print("=" * 66)
msgs = build_messages()
print(f"  共 {len(msgs)} 条，总长 {total(msgs)} 字")
print("  每一条：")
for i, m in enumerate(msgs):
    extra = "  ← 工具调用声明" if m.get("tool_calls") else \
            "  ← 工具返回结果" if m.get("role") == "tool" else ""
    print(f"    {i}. {m['role']:<9} {size(m):>4} 字{extra}")


print("\n" + "=" * 66)
print("  四种砍法对比")
print("=" * 66)

# ---- 砍法 0：你提的 —— 只砍 assistant 消息（危险！）----
m0 = copy.deepcopy(msgs)
m0 = [m for m in m0 if m["role"] != "assistant"]
report("砍法0：把 AI 的回答删掉（你提的做法）", m0,
       "⚠ 那条内容为空但带 tool_calls 的消息也被删了 → 工具结果成了孤儿")

# ---- 砍法 1：滑窗 —— 从头砍老的，保住最近 N 条 ----
m1 = copy.deepcopy(msgs)
KEEP = 5
m1 = [m1[0]] + m1[-KEEP:]        # system 永远保留 + 最近 5 条
report("砍法1：滑窗，只留 system + 最近 5 条", m1,
       "要看运气：如果 system 后面刚好是一条孤儿 tool 消息，照样坏")

# ---- 砍法 2：正确做法 —— 按「对」砍，工具调用和结果一起砍 ----
def trim_keep_pairs(msgs, keep_from_tail=6):
    """从尾部往前数，遇到 tool 消息就把它的调用声明一起带上"""
    head = msgs[0]                      # system 永远不能砍
    body = msgs[1:]
    kept = []
    for m in reversed(body):
        kept.append(m)
        if m.get("role") == "tool":
            # 把配对的 assistant 声明也一起捞回来
            cid = m.get("tool_call_id")
            for x in body:
                if any(tc["id"] == cid for tc in (x.get("tool_calls") or [])):
                    if x not in kept:
                        kept.append(x)
        if len(kept) >= keep_from_tail:
            break
    return [head] + list(reversed(kept))

m2 = trim_keep_pairs(copy.deepcopy(msgs), keep_from_tail=6)
report("砍法2：按「调用+结果」成对保留", m2,
       "✅ 正确：工具调用和它的结果要么都留，要么都砍")

# ---- 砍法 3：真正的省钱大头 —— 截断超长的工具结果 ----
def truncate_tool_results(msgs, limit=60):
    out = copy.deepcopy(msgs)
    for m in out:
        if m.get("role") == "tool" and len(m.get("content") or "") > limit:
            m["content"] = m["content"][:limit] + f"…（已截断，原 {len(m['content'])} 字）"
    return out

m3 = truncate_tool_results(copy.deepcopy(msgs), limit=60)
report("砍法3：只截断超长的工具结果（其他都不动）", m3,
       "✅ 最划算：工具结果通常是最肥的一块，且信息已被模型消化过一轮")


print("\n" + "=" * 66)
print("  结论")
print("=" * 66)
print("""
  你说的「出了结果后砍」要改成：

    ⛔ 不是「结果出来之后」
    ✅ 而是「每一圈循环开头、发给模型之前」都检查一次

  因为雪球是每一圈都在涨，不是涨一次。

  砍的顺序（按性价比排）：

    1. 截断超长工具结果   ← 最肥、最安全、不要钱
    2. 滑窗丢掉老轮次     ← 要成对砍，否则报错
    3. LLM 摘要老对话     ← 最后才用，它自己要花钱要时间

  三条红线（永远不能碰）：

    · system prompt 不能砍 —— 砍了它就像换了个人
    · 最近一轮不能砍 —— 用户刚问的话丢了，它就答非所问
    · 「工具调用声明」和「工具返回结果」必须成对 —— 砍一个留一个，
      接口直接报 400。这是真实项目里最常见的踩坑点。

  面试一句话版本：
  「上下文管理 = 截断 → 滑窗 → 摘要，三条红线是 system、最近一轮、
   以及 tool_call 与 tool 结果的配对。」
""")
