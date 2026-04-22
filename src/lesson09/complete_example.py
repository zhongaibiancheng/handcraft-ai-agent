"""
第九课完整示例 - 原子动作系统

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
from agent.agent import Agent


def lesson_09_atomic_actions():
    """第九课演示：原子动作系统"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第九课：原子动作——让每一步执行都可验证")
    print("=" * 60)
    print()
    print("📝 说明：Agent 会将模糊的计划步骤转换为带参数的原子动作。")
    print("   每个原子动作都有 action（动作名）和 inputs（参数）。")
    print()

    # ========== 场景 1：单个步骤转换 ==========
    print("=" * 60)
    print("  场景 1：将模糊步骤转为原子动作")
    print("=" * 60)
    print()

    step1 = "写一篇关于 AI Agent 的入门介绍"
    print(f"📝 模糊步骤：{step1}\n")

    atomic1 = agent.create_atomic_action(step1)

    if atomic1:
        print(f"\n📋 转换结果：")
        print(f"   动作名称：{atomic1['action']}")
        print(f"   动作参数：")
        for key, value in atomic1.get('inputs', {}).items():
            print(f"     - {key}：{value}")
    else:
        print("❌ 原子动作生成失败")

    # ========== 场景 2：不同步骤 → 不同动作类型 ==========
    print("\n\n" + "=" * 60)
    print("  场景 2：不同类型的步骤，不同的原子动作")
    print("=" * 60)
    print()

    steps = [
        "研究竞品的定价策略",
        "写一封客户跟进邮件",
        "整理上周的销售数据报表",
        "制作季度汇报 PPT",
    ]

    for step in steps:
        print(f"📝 步骤：{step}")
        atomic = agent.create_atomic_action(step)
        if atomic:
            print(f"   → {atomic['action']}({atomic.get('inputs', {})})")
        else:
            print(f"   → ❌ 转换失败")
        print()

    # ========== 场景 3：计划 → 原子动作（完整链路） ==========
    print("=" * 60)
    print("  场景 3：目标 → 计划 → 原子动作（完整链路）")
    print("=" * 60)
    print()

    goal3 = "策划一场面向初中生的 AI 编程体验课"
    print(f"🎯 目标：{goal3}\n")

    # 第一步：生成计划（第八课能力）
    print("--- 第八课：生成计划 ---")
    plan3 = agent.create_plan(goal3)

    if plan3:
        print("\n📋 计划内容：")
        for i, step in enumerate(plan3["steps"], 1):
            print(f"  {i}. {step}")

        # 第二步：把每步转为原子动作（第九课能力）
        print("\n--- 第九课：转换为原子动作 ---")
        atomic_actions = agent.convert_plan_to_atomic_actions(plan3)

        if atomic_actions:
            print("\n📊 步骤 → 动作映射：")
            for a in atomic_actions:
                num = a.get("step_num", "?")
                original = a.get("original_step", "?")
                action = a.get("action", "?")
                inputs = a.get("inputs", {})
                converted = "✅" if a.get("converted", True) else "⚠️"
                print(f"  {converted} 步骤{num}：{original}")
                print(f"         → {action}({inputs})")
                print()
    else:
        print("❌ 计划生成失败")

    # ========== 场景 4：验证原子动作 ==========
    print("=" * 60)
    print("  场景 4：验证原子动作的完整性")
    print("=" * 60)
    print()

    step4 = "查询深圳今天的天气情况"
    print(f"📝 步骤：{step4}\n")

    atomic4 = agent.create_atomic_action(step4)

    if atomic4:
        print("🔍 验证结果：")

        # 检查 action 字段
        has_action = "action" in atomic4 and atomic4["action"]
        print(f"  {'✅' if has_action else '❌'} action 字段：{'存在' if has_action else '缺失'}")

        # 检查 inputs 字段
        has_inputs = "inputs" in atomic4 and isinstance(atomic4["inputs"], dict)
        print(f"  {'✅' if has_inputs else '❌'} inputs 字段：{'存在且为字典' if has_inputs else '缺失或格式错误'}")

        # 检查 inputs 非空
        inputs_not_empty = len(atomic4.get("inputs", {})) > 0
        print(f"  {'✅' if inputs_not_empty else '⚠️'} inputs 内容：{'非空' if inputs_not_empty else '为空（建议优化 Prompt）'}")

        # 打印完整内容
        print(f"\n📋 完整原子动作：")
        print(f"   {atomic4}")
    else:
        print("❌ 原子动作生成失败")

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
    """交互模式：用户输入步骤，Agent 转换为原子动作"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第九课：原子动作系统 - 交互模式")
    print("=" * 60)
    print()
    print("💬 输入一个计划步骤，Agent 会将其转换为原子动作。")
    print("   输入 'quit' 或 'exit' 退出。")
    print("   输入 'plan' 可以先生成一个完整计划再转换。")
    print("   输入 'state' 查看当前状态。")
    print()

    while True:
        try:
            user_input = input("📝 步骤/目标: ").strip()
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

        if user_input.lower() == "plan":
            try:
                goal = input("🎯 请输入目标: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if goal:
                print()
                plan = agent.create_plan(goal)
                if plan:
                    print()
                    actions = agent.convert_plan_to_atomic_actions(plan)
            continue

        print()

        # 单个步骤 → 原子动作
        atomic = agent.create_atomic_action(user_input)

        if atomic:
            print(f"\n📋 原子动作详情：")
            print(f"   动作名称：{atomic['action']}")
            print(f"   动作参数：")
            for key, value in atomic.get('inputs', {}).items():
                print(f"     - {key}：{value}")
        else:
            print("❌ 原子动作生成失败，请重试。")

        print()


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_09_atomic_actions()


if __name__ == "__main__":
    main()
