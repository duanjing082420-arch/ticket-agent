"""
不带 AI 的纯循环演示：看 continue 到底做了什么

这个脚本不联网、不花钱。
我用三句「假装的模型回答」代替真实模型，目的只有一个：
让你看清 while + continue 是怎么做到「连调多次工具」的。

口诀：
    执行工具 = continue 上面那几行
    continue  = 回到 while 那一行，再问一次模型
"""

# 假装这是模型每一轮的回答（真实情况由 API 返回）
fake_model_answers = [
    "我要调工具：get_order_status",
    "我要调工具：get_tracking_detail",
    "我不需要工具了，这是我的最终回答",
]

i = 0
step = 1

while step <= 5:                              # ← 第 22 行：圈子的入口
    print(f"\n========== 第 {step} 圈 ==========")

    answer = fake_model_answers[i]             # ← 真实代码这里是「调 API 问模型」
    i += 1
    print(f"  问模型 → 它说：{answer}")
    step += 1

    if "我要调工具" in answer:
        print("  ├─ 有 tool_calls → 执行工具（查文件）")
        print("  └─ 把结果塞进 messages，然后 continue")
        continue                               # ← 第 34 行：立刻回到第 22 行

    # ↓↓↓ 只有「模型没要工具」时才会走到这里 ↓↓↓
    print("  ├─ 没有 tool_calls → 打印回答")
    print("  └─ 等用户说话（或 break）")
    break

print(f"\n一共转了 {step - 1} 圈")
print()
print("看明白了吗：continue 一次工具都没调，它只是让循环再转一圈。")
print("真正执行工具的，是 continue 上面那两行。")
print("每一圈都会重新问一次模型 —— 所以是「问→查→问→查→问→答」。")
