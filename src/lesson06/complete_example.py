"""
第六课完整示例 - 智能体循环

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
from agent.agent import Agent


def lesson_06_agent_loop():
    """第六课演示：智能体循环"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第六课：智能体循环")
    print("=" * 60)
    print()
    print("📝 说明：Agent 会反复思考和行动，直到任务完成或达到最大步数。")
    print("   早期步骤中的重复分析是正常的——Agent 在逐步完善理解。")
    print()

    # 演示 1：基础循环（3步）
    print("=" * 60)
    print("  演示 1：基础循环（max_steps=3）")
    print("=" * 60)
    print()

    results = agent.run_loop("帮我分析一下 Python 的优缺点", max_steps=3)

    print("\n--- 结果汇总 ---")
    for i, result in enumerate(results, 1):
        action = result.get("action", "unknown")
        reason = result.get("reason", "无原因")
        print(f"  步骤{i}: [{action}] {reason}")

    # 演示 2：更大步数
    print("\n\n" + "=" * 60)
    print("  演示 2：更大步数（max_steps=5）")
    print("=" * 60)
    print()

    results = agent.run_loop("帮我研究一下深度学习和传统机器学习的区别", max_steps=5)

    print("\n--- 结果汇总 ---")
    for i, result in enumerate(results, 1):
        action = result.get("action", "unknown")
        reason = result.get("reason", "无原因")
        print(f"  步骤{i}: [{action}] {reason}")

    # 演示 3：简单问题（应该 1-2 步就 done）
    print("\n\n" + "=" * 60)
    print("  演示 3：简单问题（预期快速完成）")
    print("=" * 60)
    print()

    results = agent.run_loop("你好", max_steps=5)

    print("\n--- 结果汇总 ---")
    for i, result in enumerate(results, 1):
        action = result.get("action", "unknown")
        reason = result.get("reason", "无原因")
        print(f"  步骤{i}: [{action}] {reason}")

    print("\n\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


def interactive_mode():
    """交互模式：用户输入，Agent 循环处理"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第六课：智能体循环 - 交互模式")
    print("=" * 60)
    print()
    print("输入你的问题，Agent 会循环处理。")
    print("输入 'quit' 或 'exit' 退出。")
    print("输入 'state' 查看当前状态。")
    print()

    while True:
        try:
            user_input = input("💬 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "state":
            agent.show_state()
            continue

        if user_input.lower() == "info":
            agent.show_info()
            continue

        # 获取 max_steps
        max_steps_input = input("🔢 最大步数（默认3）: ").strip()
        try:
            max_steps = int(max_steps_input) if max_steps_input else 3
        except ValueError:
            max_steps = 3

        print()
        results = agent.run_loop(user_input, max_steps=max_steps)

        print("\n--- 结果汇总 ---")
        for i, result in enumerate(results, 1):
            action = result.get("action", "unknown")
            reason = result.get("reason", "无原因")
            print(f"  步骤{i}: [{action}] {reason}")

        print()


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_06_agent_loop()


if __name__ == "__main__":
    main()
