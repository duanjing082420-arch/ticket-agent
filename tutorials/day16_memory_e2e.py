"""day16：记忆端到端验证 —— 通过真实 service.py 管线，证明跨轮记忆生效。

运行：.venv/Scripts/python.exe tutorials/day16_memory_e2e.py

设计：同一 user_id 连续两轮，且两轮都走「主流程」（faq/query/action 意图），
因为只有主流程才会注入历史 + 落盘。
  轮1：说出订单号 SO20240905004（存进记忆）
  轮2：故意不写订单号，用「这个订单」指代——模型只能靠注入的历史才知道是哪个。
若轮2 回答里出现 SO20240905004，说明长期记忆把轮1 接上了。

注意：chitchat / unknown 是短路分支，不进主流程，目前不触发记忆（见末尾说明）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import Settings, load_dotenv
from agent.service import TicketAgent

UID = "demo_mem_001"


def main() -> None:
    load_dotenv()
    agent = TicketAgent(Settings.from_env())

    # 跑前清空该用户，保证可重复
    deleted = agent.memory.clear(UID)
    print(f"[准备] 清空 {UID} 的历史 {deleted} 条\n")

    # 轮1：在主流程里说出订单号（会被存进记忆）
    r1 = agent.handle("帮我查一下订单 SO20240905004 的物流到哪了", user_id=UID)
    print("轮1 用户: 帮我查一下订单 SO20240905004 的物流到哪了")
    print("轮1 助手:", r1.output[:160].replace("\n", " "), "\n")

    # 轮2：故意不写订单号，用「这个订单」指代，模型只能靠记忆里的历史确认是哪个
    r2 = agent.handle("这个订单的物流现在到哪一步了？", user_id=UID)
    print("轮2 用户: 这个订单的物流现在到哪一步了？")
    print("轮2 助手:", r2.output[:240].replace("\n", " "), "\n")

    ok = "SO20240905004" in r2.output
    print(
        "记忆生效判定:",
        "✅ PASS（轮2 没写订单号却答出了 SO20240905004，来自注入的历史）"
        if ok
        else "⚠️ 轮2 未引用订单号，可能模型改口追问，请核对上面回答",
    )
    print(
        f'\n[查看落盘] 可运行：sqlite3 data/memory.db '
        f'"SELECT role,content FROM memory_turns WHERE user_id=\\"{UID}\\";"'
    )


if __name__ == "__main__":
    main()
