"""
第七课完整示例 - 记忆系统

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
from agent.agent import Agent


def lesson_07_memory():
    """第七课演示：记忆系统"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第七课：记忆系统")
    print("=" * 60)
    print()
    print("📝 说明：Agent 会在对话中自动记住重要信息。")
    print("   多次独立对话之间共享记忆。")
    print()

    # 对话 1：告诉 Agent 你的名字和职业
    print("=" * 60)
    print("  对话 1：告诉 Agent 你的名字和职业")
    print("=" * 60)
    print()

    r1 = agent.run_with_memory("我叫小明，是一名后端开发工程师")
    if r1:
        print(f"🤖 Agent: {r1['reply']}")
    else:
        print("❌ 对话失败")

    # 对话 2：问它记不记得
    print("\n\n" + "=" * 60)
    print("  对话 2：问它记不记得你的名字")
    print("=" * 60)
    print()

    r2 = agent.run_with_memory("你还记得我的名字吗？")
    if r2:
        print(f"🤖 Agent: {r2['reply']}")
    else:
        print("❌ 对话失败")

    # 对话 3：告诉更多偏好信息
    print("\n\n" + "=" * 60)
    print("  对话 3：告诉 Agent 你的技术栈偏好")
    print("=" * 60)
    print()

    r3 = agent.run_with_memory("我主要用 Python 和 Go，最近在学 AI Agent 开发")
    if r3:
        print(f"🤖 Agent: {r3['reply']}")
    else:
        print("❌ 对话失败")

    # 对话 4：综合提问（需要结合多条记忆）
    print("\n\n" + "=" * 60)
    print("  对话 4：综合提问（需要结合多条记忆）")
    print("=" * 60)
    print()

    r4 = agent.run_with_memory("帮我推荐一个学习路线，结合我的背景和技术栈")
    if r4:
        print(f"🤖 Agent: {r4['reply']}")
    else:
        print("❌ 对话失败")

    # 对话 5：不需要记忆的普通问题
    print("\n\n" + "=" * 60)
    print("  对话 5：普通问题（不应产生新记忆）")
    print("=" * 60)
    print()

    r5 = agent.run_with_memory("帮我写一个冒泡排序")
    if r5:
        print(f"🤖 Agent: {r5['reply']}")
    else:
        print("❌ 对话失败")

    # 查看所有记忆
    print("\n\n" + "=" * 60)
    print("  🧠 最终记忆状态")
    print("=" * 60)
    print()

    agent.memory.show()

    # 演示记忆搜索
    print("\n🔍 搜索 'Python'：", agent.memory.search("Python"))
    print("🔍 搜索 '前端'：", agent.memory.search("前端"))

    # 演示记忆删除
    print("\n🗑️ 删除一条记忆...")
    agent.memory.remove(agent.memory.get_all()[0] if agent.memory.get_all() else "不存在")
    print(f"删除后剩余 {agent.memory.count()} 条记忆")

    print("\n\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


def interactive_mode():
    """交互模式：用户输入，Agent 带记忆回复"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第七课：记忆系统 - 交互模式")
    print("=" * 60)
    print()
    print("💬 和 Agent 对话，它会自动记住重要信息。")
    print("   输入 'quit' 或 'exit' 退出。")
    print("   输入 'memory' 查看所有记忆。")
    print("   输入 'search 关键词' 搜索记忆。")
    print("   输入 'clear_memory' 清空记忆。")
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

        if user_input.lower() == "memory":
            agent.memory.show()
            continue

        if user_input.lower() == "clear_memory":
            agent.memory.clear()
            continue

        if user_input.lower().startswith("search "):
            keyword = user_input[7:].strip()
            results = agent.memory.search(keyword)
            if results:
                print(f"🔍 搜索 '{keyword}' 的结果：")
                for r in results:
                    print(f"  - {r}")
            else:
                print(f"🔍 没有找到包含 '{keyword}' 的记忆")
            continue

        print()
        response = agent.run_with_memory(user_input)
        if response:
            print(f"🤖 Agent: {response['reply']}")
        else:
            print("❌ 对话失败")

        print()


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_07_memory()


if __name__ == "__main__":
    main()
