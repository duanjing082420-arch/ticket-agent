"""day15：记忆层 demo —— 不依赖 LLM，纯测 SQLite 存取与历史拼装。

运行：.venv/Scripts/python.exe tutorials/day15_memory_test.py

目的：在把它接进 service.py 之前，先孤立验证 MemoryStore 的存取逻辑，
避免「还没搞懂零件就装上车」——这也是排查复杂系统的常用手法。
"""

import os
import sys
import tempfile

# 把项目根加进 import 路径（tutorials/ 下的脚本统一靠这个找到 agent 包）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory import MemoryStore


def main() -> None:
    # 用系统临时目录的库，避免污染真实的 data/memory.db
    tmp_db = os.path.join(tempfile.gettempdir(), "day15_mem_test.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    store = MemoryStore(tmp_db)

    uid = "tester_001"

    # 模拟两轮对话：用户逐步补充信息
    store.save_turn(uid, "我是会员", "好的，已记录您是会员用户。")
    store.save_turn(
        uid,
        "我的会员券能叠加吗",
        "会员券能否叠加取决于具体活动规则，我帮您查一下可叠加的券……",
    )

    hist = store.load_history(uid, limit=6)
    print("load_history 取出的历史（按时间正序）：")
    for m in hist:
        print(f"  [{m['role']:>9}] {m['content']}")

    # 演示：如何把历史拼进 OpenAI messages
    # 顺序很关键：system → 历史(早→晚) → 当前新问题（最后一条）
    messages = [{"role": "system", "content": "你是售后助手"}]
    messages.extend(hist)
    messages.append({"role": "user", "content": "那运费券也能叠加吗？"})

    print("\n拼好的 messages（最后一条是当前新问题，前面是记忆）：")
    for m in messages:
        print(f"  {m['role']:>9}: {m['content'][:34]}")

    # 清理测试数据
    deleted = store.clear(uid)
    print(f"\n已清空测试用户历史（删除 {deleted} 条）。临时库：{tmp_db}")


if __name__ == "__main__":
    main()
