"""
完整 CLI 交互程序 - 第五课
支持：工具调用演示 + 连续计算交互 + 多工具测试

依赖：pip install openai
"""

import sys
import time
from agent.agent import Agent, calculator


def print_separator() -> None:
    """打印分隔线"""
    print("\n" + "─" * 55)


def demo_mode(agent: Agent) -> None:
    """演示模式：自动展示工具调用效果"""
    print("\n🎭 ====== 演示模式 ======\n")
    print("接下来会自动测试多个数学问题，展示 AI 如何识别意图、选择工具、提取参数。\n")

    test_cases = [
        ("42 * 7 等于多少", "乘法"),
        ("100 减去 37", "减法"),
        ("25 除以 5", "除法"),
        ("What is 128 + 256?", "加法（英文输入）"),
        ("3.14 乘以 2", "小数运算"),
    ]

    for i, (user_input, label) in enumerate(test_cases, 1):
        print_separator()
        print(f"测试 {i}/{len(test_cases)}：{label}")
        print(f"💬 用户：{user_input}")
        print()

        start = time.time()

        # 第一步：AI 请求工具
        tool_call = agent.request_tool(user_input)

        if tool_call:
            print(f"🔧 AI 请求：{tool_call}")
            print()

            # 第二步：代码执行工具
            result = agent.execute_tool_call(tool_call)
            print(f"📊 执行结果：{result}")
        else:
            print("❌ 工具调用失败：AI 无法理解请求")

        elapsed = time.time() - start
        print(f"\n⏱ 耗时：{elapsed:.1f}秒")
        time.sleep(1)

    print_separator()
    print("\n✅ 演示完成！")
    print("\n💡 你看到了：")
    print("   - AI 不做数学运算，它只负责「选工具 + 提参数」")
    print("   - 真正的计算由你的代码完成")
    print("   - 请求和执行是分开的——这是安全性的基础\n")


def interactive_mode(agent: Agent) -> None:
    """交互模式：连续计算对话"""
    print("\n💬 ====== 交互模式 ======\n")

    agent.show_tools()

    print("\n使用说明：")
    print("  - 输入数学表达式，如「42 * 7」或「100 减去 37」")
    print("  - 输入「工具」查看可用工具")
    print("  - 输入「退出」结束程序")
    print_separator()

    while True:
        try:
            user_input = input("\n你：").strip()

            if not user_input:
                continue

            if user_input in ("退出", "exit", "quit", "q"):
                print("\n👋 再见！\n")
                break

            elif user_input == "工具":
                agent.show_tools()
                continue

            # 工具调用
            tool_call = agent.request_tool(user_input)

            if tool_call:
                print(f"\n🔧 工具调用：{tool_call}")
                result = agent.execute_tool_call(tool_call)
                print(f"📊 结果：{result}")
            else:
                print("\n🤔 抱歉，我无法理解你的请求。试试输入一个数学运算？")

        except KeyboardInterrupt:
            print("\n\n👋 再见！\n")
            break
        except Exception as e:
            print(f"\n⚠️ 出错了：{e}")


def multi_tool_demo(agent: Agent) -> None:
    """多工具演示：展示工具注册和选择的灵活性"""
    print("\n🔧 ====== 多工具演示 ======\n")

    # 定义更多工具
    def string_reverse(text: str) -> str:
        """反转字符串"""
        return text[::-1]

    def string_length(text: str) -> int:
        """计算字符串长度"""
        return len(text)

    def string_upper(text: str) -> str:
        """转大写"""
        return text.upper()

    def string_concat(a: str, b: str) -> str:
        """拼接字符串"""
        return a + b

    # 注册字符串工具
    agent.register_tool(
        name="string_reverse",
        func=string_reverse,
        description="反转字符串",
        parameters={
            "text": {"type": "string", "description": "要反转的字符串", "required": True},
        },
    )
    agent.register_tool(
        name="string_length",
        func=string_length,
        description="计算字符串的长度（字符数）",
        parameters={
            "text": {"type": "string", "description": "要计算的字符串", "required": True},
        },
    )
    agent.register_tool(
        name="string_upper",
        func=string_upper,
        description="将字符串转为大写",
        parameters={
            "text": {"type": "string", "description": "要转换的字符串", "required": True},
        },
    )
    agent.register_tool(
        name="string_concat",
        func=string_concat,
        description="拼接两个字符串",
        parameters={
            "a": {"type": "string", "description": "第一个字符串", "required": True},
            "b": {"type": "string", "description": "第二个字符串", "required": True},
        },
    )

    agent.show_tools()

    # 测试：AI 自动选择合适的工具
    test_cases = [
        ("把 hello 反转一下", "string_reverse"),
        ("hello world 有多少个字符", "string_length"),
        ("把 hello 变成大写", "string_upper"),
        ("把 hello 和 world 拼在一起", "string_concat"),
    ]

    for user_input, expected_tool in test_cases:
        print_separator()
        print(f"💬 用户：{user_input}")
        print(f"🎯 期望工具：{expected_tool}")

        tool_call = agent.request_tool(user_input)

        if tool_call:
            actual_tool = tool_call.get("tool", "N/A")
            match = "✅" if actual_tool == expected_tool else "❌"
            print(f"{match} 实际工具：{actual_tool}")
            print(f"🔧 参数：{tool_call.get('arguments', {})}")

            result = agent.execute_tool_call(tool_call)
            print(f"📊 结果：{result}")
        else:
            print("❌ 工具调用失败")

    print_separator()
    print("\n✅ 多工具演示完成！")
    print("\n💡 你看到了：")
    print("   - 添加新工具只需调用 register_tool()，无需修改模型")
    print("   - AI 能从多个工具中自动选择最合适的")
    print("   - 每个工具的参数由你的代码定义和控制\n")


def main():
    """主入口"""
    print("""
╔══════════════════════════════════════════════╗
║   🔧 AI Agent 工具调用 — 第五课演示         ║
║   手搓 AI Agent 从 0 到 1                   ║
╚══════════════════════════════════════════════╝
    """)

    # 初始化 Agent
    model = "qwen2.5:7b"
    if len(sys.argv) > 1:
        model = sys.argv[1]

    print(f"📦 正在加载模型：{model} ...")
    agent = Agent(model=model)
    print("✅ 模型加载完成！\n")

    # 注册默认工具
    agent.register_tool(
        name="calculator",
        func=calculator,
        description="数学计算器，支持加减乘除",
        parameters={
            "a": {"type": "number", "description": "第一个数字", "required": True},
            "b": {"type": "number", "description": "第二个数字", "required": True},
            "operation": {
                "type": "string",
                "description": "运算类型：add（加）、subtract（减）、multiply（乘）、divide（除）",
                "required": True,
            },
        },
    )

    # 选择模式
    if "--demo" in sys.argv:
        demo_mode(agent)
    elif "--multi" in sys.argv:
        multi_tool_demo(agent)
    else:
        print("选择模式：")
        print("  1. 🎭 演示模式（自动展示工具调用效果）")
        print("  2. 💬 交互模式（手动输入计算）")
        print("  3. 🔧 多工具演示（展示多工具选择能力）")
        print()

        choice = input("请选择（1/2/3，默认2）：").strip()

        if choice == "1":
            demo_mode(agent)
        elif choice == "3":
            multi_tool_demo(agent)
        else:
            interactive_mode(agent)


if __name__ == "__main__":
    main()
