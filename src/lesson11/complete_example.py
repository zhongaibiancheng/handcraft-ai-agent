"""
第十一课完整示例 - 评测系统（智能体的回归测试）

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
from agent.agent import Agent, calculator


def lesson_11_evals():
    """第十一课演示：评测系统"""

    agent = Agent(model="qwen2.5:7b")

    # 注册工具（评测需要）
    agent.register_tool(
        name="calculator",
        func=calculator,
        description="数学计算器，支持加减乘除运算",
        parameters={
            "a": {"type": "number", "description": "第一个数", "required": True},
            "b": {"type": "number", "description": "第二个数", "required": True},
            "operation": {"type": "string", "description": "运算类型", "required": True},
        }
    )

    print("=" * 60)
    print("  第十一课：评测——给 Agent 装上回归测试")
    print("=" * 60)
    print()
    print("📝 说明：评测套件会运行 Agent 的三个核心能力，")
    print("   验证结构化输出、工具调用和记忆是否正常工作。")
    print()

    # ========== 场景 1：全量评测 ==========

    print("=" * 60)
    print("  场景 1：全量评测（结构化输出 + 工具调用 + 记忆）")
    print("=" * 60)
    print()

    from agent.evals import AgentEval, print_eval_report
    from evals.golden_datasets import (
        STRUCTURED_OUTPUT_GOLDEN,
        TOOL_CALL_GOLDEN,
        MEMORY_GOLDEN
    )

    evaluator = AgentEval(agent)

    total_cases = len(STRUCTURED_OUTPUT_GOLDEN) + len(TOOL_CALL_GOLDEN) + len(MEMORY_GOLDEN)
    print(f"🧪 总共 {total_cases} 个用例\n")

    results = evaluator.run_all(
        structured_cases=STRUCTURED_OUTPUT_GOLDEN,
        tool_cases=TOOL_CALL_GOLDEN,
        memory_cases=MEMORY_GOLDEN
    )

    print_eval_report(results)

    # ========== 场景 2：单套件评测 ==========

    print("\n\n" + "=" * 60)
    print("  场景 2：仅结构化输出测试")
    print("=" * 60)
    print()

    results_structured = evaluator.test_structured_output(STRUCTURED_OUTPUT_GOLDEN)
    print_eval_report([results_structured])

    # ========== 场景 3：捕获回归（修改 System Prompt） ==========

    print("\n\n" + "=" * 60)
    print("  场景 3：捕获回归 —— 修改 System Prompt 后重新评测")
    print("=" * 60)
    print()
    print("📝 模拟修改：给 System Prompt 加一段引导性描述")
    print()

    # 保存原始 Prompt
    original_prompt = agent.system_prompt

    # 修改 System Prompt（可能破坏结构化输出）
    agent.system_prompt = "你是一个友好的AI助手。在回答之前，请先简要解释你的思考过程。"

    print(f"  原始 Prompt：{original_prompt[:30]}...")
    print(f"  修改后 Prompt：{agent.system_prompt[:40]}...")
    print()

    # 用修改后的 Prompt 重新评测结构化输出
    results_modified = evaluator.test_structured_output(STRUCTURED_OUTPUT_GOLDEN)
    print_eval_report([results_modified])

    if results_modified.failed > 0:
        print("\n⚠️  修改 System Prompt 后，部分用例失败了！")
        print("   这就是评测的价值 —— 在部署前捕获回归。")
    else:
        print("\n✅ 修改后所有用例仍然通过。")

    # 恢复原始 Prompt
    agent.system_prompt = original_prompt

    # ========== 场景 4：使用 Agent 的便捷方法 ==========

    print("\n\n" + "=" * 60)
    print("  场景 4：使用 Agent.run_evals() 便捷方法")
    print("=" * 60)
    print()

    print("📝 Agent.run_evals() = 一键全量评测")
    print("   Agent.run_quick_eval() = 一键快速评测（仅结构化输出）")
    print()

    agent.run_quick_eval()

    # 查看最终状态
    print("\n\n" + "=" * 60)
    print("  📊 最终 Agent 状态")
    print("=" * 60)
    print()
    agent.show_info()

    print("\n\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


def interactive_mode():
    """交互模式：运行评测或对话"""

    agent = Agent(model="qwen2.5:7b")

    # 注册工具
    agent.register_tool(
        name="calculator",
        func=calculator,
        description="数学计算器，支持加减乘除运算",
        parameters={
            "a": {"type": "number", "description": "第一个数", "required": True},
            "b": {"type": "number", "description": "第二个数", "required": True},
            "operation": {"type": "string", "description": "运算类型", "required": True},
        }
    )

    print("=" * 60)
    print("  第十一课：评测系统 - 交互模式")
    print("=" * 60)
    print()
    print("命令：")
    print("  eval     - 运行全量评测")
    print("  quick    - 运行快速评测")
    print("  chat     - 进入对话模式")
    print("  state    - 查看 Agent 状态")
    print("  quit     - 退出")
    print()

    in_chat = False

    while True:
        try:
            if in_chat:
                user_input = input("💬 你: ").strip()
            else:
                user_input = input("🎯 命令: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if in_chat:
            if user_input.lower() in ("quit", "exit", "/quit"):
                in_chat = False
                print("\n已退出对话模式。")
                continue

            reply = agent.chat(user_input)
            print(f"🤖 Agent: {reply}")
            continue

        # 非对话模式命令处理
        if user_input.lower() == "eval":
            agent.run_evals()
        elif user_input.lower() == "quick":
            agent.run_quick_eval()
        elif user_input.lower() == "chat":
            in_chat = True
            print("\n进入对话模式（输入 /quit 退出）：\n")
        elif user_input.lower() == "state":
            agent.show_info()
        else:
            print("❌ 未知命令。可用命令：eval, quick, chat, state, quit")


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_11_evals()


if __name__ == "__main__":
    main()
