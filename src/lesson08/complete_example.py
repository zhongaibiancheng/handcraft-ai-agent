"""
第八课完整示例 - 规划系统

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
from agent.agent import Agent


def lesson_08_planning():
    """第八课演示：规划系统"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第八课：规划系统——先想后做")
    print("=" * 60)
    print()
    print("📝 说明：Agent 会先为目标生成执行计划，然后按计划逐步执行。")
    print("   规划与执行分离——你可以在执行前审查和修改计划。")
    print()

    # ========== 场景 1：基础规划并执行 ==========
    print("=" * 60)
    print("  场景 1：生成计划并直接执行")
    print("=" * 60)
    print()

    goal1 = "写一篇关于 AI Agent 技术的入门博客"
    print(f"🎯 目标：{goal1}\n")

    plan1 = agent.create_plan(goal1)

    if plan1:
        # 展示计划
        print("\n📋 AI 生成的计划：")
        for i, step in enumerate(plan1["steps"], 1):
            print(f"  {i}. {step}")

        # 直接执行
        results1 = agent.execute_plan(plan1)
        print(f"📊 执行完成，共 {len(results1)} 个步骤")
    else:
        print("❌ 计划生成失败")

    # ========== 场景 2：规划但不执行（审查模式） ==========
    print("\n\n" + "=" * 60)
    print("  场景 2：生成计划 → 人工审查 → 决定是否执行")
    print("=" * 60)
    print()

    goal2 = "策划一场 Python 技术分享会"
    print(f"🎯 目标：{goal2}\n")

    plan2 = agent.create_plan(goal2)

    if plan2:
        print("\n📋 AI 生成的计划：")
        for i, step in enumerate(plan2["steps"], 1):
            print(f"  {i}. {step}")

        print("\n💡 演示中自动选择执行。实际使用时，你可以在这里审查计划，")
        print("   修改、删除、添加步骤，然后再决定是否执行。")

        # 直接执行（演示用）
        results2 = agent.execute_plan(plan2)
        print(f"📊 执行完成，共 {len(results2)} 个步骤")
    else:
        print("❌ 计划生成失败")

    # ========== 场景 3：修改计划再执行 ==========
    print("\n\n" + "=" * 60)
    print("  场景 3：生成计划 → 手动修改 → 执行修改后的计划")
    print("=" * 60)
    print()

    goal3 = "搭建一个个人博客网站"
    print(f"🎯 目标：{goal3}\n")

    plan3 = agent.create_plan(goal3)

    if plan3:
        print("\n📋 AI 原始计划：")
        for i, step in enumerate(plan3["steps"], 1):
            print(f"  {i}. {step}")

        # 手动修改：在开头添加一步
        print("\n✏️ 手动修改：在开头添加"调研建站方案"...")
        plan3["steps"].insert(0, "调研主流建站方案（静态站点/WordPress/Hugo）并选型")

        # 手动修改：限制最多 5 步
        if len(plan3["steps"]) > 5:
            print(f"✏️ 手动修改：步骤从 {len(plan3['steps'])} 步精简到 5 步...")
            plan3["steps"] = plan3["steps"][:5]

        print("\n📋 修改后的计划：")
        for i, step in enumerate(plan3["steps"], 1):
            print(f"  {i}. {step}")

        results3 = agent.execute_plan(plan3)
        print(f"📊 执行完成，共 {len(results3)} 个步骤")
    else:
        print("❌ 计划生成失败")

    # ========== 场景 4：相同目标多次规划 ==========
    print("\n\n" + "=" * 60)
    print("  场景 4：相同目标生成两次计划，观察差异")
    print("=" * 60)
    print()

    goal4 = "学习一门新的编程语言"
    print(f"🎯 目标：{goal4}\n")

    print("--- 第一次规划 ---")
    plan4a = agent.create_plan(goal4)
    if plan4a:
        for i, step in enumerate(plan4a["steps"], 1):
            print(f"  {i}. {step}")

    print("\n--- 第二次规划 ---")
    plan4b = agent.create_plan(goal4)
    if plan4b:
        for i, step in enumerate(plan4b["steps"], 1):
            print(f"  {i}. {step}")

    print("\n💡 注意：由于 LLM 有一定的随机性，两次生成的计划可能不同。")
    print("   temperature=0.0 能减少差异，但不能完全消除。")

    # 查看最终状态
    print("\n\n" + "=" * 60)
    print("  📊 最终 Agent 状态")
    print("=" * 60)
    print()
    agent.show_state()

    print("\n\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


def interactive_mode():
    """交互模式：用户输入目标，Agent 生成计划并执行"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第八课：规划系统 - 交互模式")
    print("=" * 60)
    print()
    print("💬 输入一个目标，Agent 会生成执行计划。")
    print("   输入 'quit' 或 'exit' 退出。")
    print("   输入 'state' 查看当前状态。")
    print("   输入 'memory' 查看记忆。")
    print()

    while True:
        try:
            user_input = input("🎯 目标: ").strip()
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

        if user_input.lower() == "memory":
            agent.memory.show()
            continue

        print()

        # 生成计划
        plan = agent.create_plan(user_input)

        if plan:
            # 展示计划
            print("\n📋 计划内容：")
            for i, step in enumerate(plan["steps"], 1):
                print(f"  {i}. {step}")

            # 询问是否执行
            print()
            try:
                confirm = input("🚀 执行这个计划吗？(y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if confirm in ("y", "yes"):
                results = agent.execute_plan(plan)
                print(f"\n📊 执行完成，共 {len(results)} 个步骤")
            else:
                print("⏸️ 已取消执行。计划保存在 Agent 状态中。")
        else:
            print("❌ 计划生成失败，请重试。")

        print()


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_08_planning()


if __name__ == "__main__":
    main()
