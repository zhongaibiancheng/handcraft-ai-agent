"""
第十二课完整示例 - 遥测系统（运行时可观测性）

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
import time
from agent.agent import Agent, calculator
from agent.telemetry import Telemetry, trace


def lesson_12_telemetry():
    """第十二课演示：遥测系统"""

    # ================================================================
    # 场景 1：手动遥测 —— 不依赖 Agent，直接使用 Telemetry 类
    # ================================================================

    print("=" * 60)
    print("  第十二课：遥测——运行时可观测性")
    print("=" * 60)
    print()
    print("📝 说明：遥测系统记录 LLM 调用、工具调用、记忆操作等事件，")
    print("   通过追踪 ID 链接相关事件，聚合指标让你一眼看出系统健康状态。")
    print()

    print("=" * 60)
    print("  场景 1：手动遥测——直接使用 Telemetry 类")
    print("=" * 60)
    print()
    print("📝 先不依赖 Agent，看看 Telemetry 类如何独立工作。")
    print()

    telemetry = Telemetry("demo_scene1.jsonl")

    # 开始追踪
    trace_id = telemetry.start_trace()
    print(f"🔖 开始追踪：{trace_id}")
    print()

    # 模拟几个操作
    print("模拟 LLM 调用...")
    telemetry.log_llm_call(
        prompt_length=85, response_length=230, duration_ms=1523, success=True
    )

    telemetry.log_llm_call(
        prompt_length=120, response_length=45, duration_ms=2876, success=True
    )

    print("模拟工具调用...")
    telemetry.log_tool_request("calculator", {"a": 42, "b": 7, "operation": "multiply"}, duration_ms=45)
    telemetry.log_tool_execution("calculator", 294, duration_ms=3, success=True)

    print("模拟记忆操作...")
    telemetry.log_memory_op("add", "用户偏好：Python 开发者")

    print("模拟决策...")
    telemetry.log_decision(
        ["analyze", "research", "summarize", "answer", "done"],
        "analyze", duration_ms=312
    )

    # 打印摘要
    print()
    telemetry.print_summary()

    # 打印追踪详情
    print(f"\n🔍 追踪 {trace_id} 的详细信息：")
    telemetry.print_trace_detail(trace_id)

    telemetry.clear_log()

    # ================================================================
    # 场景 2：仪表化 Agent —— 自动记录遥测
    # ================================================================

    print("\n\n" + "=" * 60)
    print("  场景 2：仪表化 Agent —— 方法自动记录遥测")
    print("=" * 60)
    print()
    print("📝 Agent 的 generate_structured()、request_tool()、chat() 等方法")
    print("   已自动注入遥测记录——调用它们会自动记录时间、长度、结果。")
    print()

    agent2 = Agent(model="qwen2.5:7b", log_file="demo_scene2.jsonl")

    # 注册工具
    agent2.register_tool(
        name="calculator",
        func=calculator,
        description="数学计算器，支持加减乘除运算",
        parameters={
            "a": {"type": "number", "description": "第一个数", "required": True},
            "b": {"type": "number", "description": "第二个数", "required": True},
            "operation": {"type": "string", "description": "运算类型", "required": True},
        }
    )

    print("\n--- 1/3：结构化输出 ---")
    result1 = agent2.generate_structured(
        "什么是 Python？",
        '{"answer": "string", "features": ["string"]}'
    )
    print(f"   结果：{result1}")

    print("\n--- 2/3：工具调用 ---")
    tool_call = agent2.request_tool("请计算 42 * 7")
    if tool_call:
        print(f"   工具请求：{tool_call}")
        result2 = agent2.execute_tool_call(tool_call)
        print(f"   执行结果：{result2}")

    print("\n--- 3/3：简单对话 ---")
    reply = agent2.chat("你好，我叫小明")
    print(f"   回复：{reply[:50]}...")

    # 查看遥测摘要
    print()
    agent2.show_telemetry()

    # 查看追踪
    if agent2.telemetry and agent2.telemetry.current_trace_id:
        print(f"\n🔍 当前追踪 ID：{agent2.telemetry.current_trace_id}")
        agent2.analyze_trace()

    agent2.telemetry.clear_log()

    # ================================================================
    # 场景 3：追踪调试 —— 找到失败的 Span
    # ================================================================

    print("\n\n" + "=" * 60)
    print("  场景 3：追踪调试——通过 trace_id 定位问题")
    print("=" * 60)
    print()
    print("📝 模拟一个失败的交互，然后通过 trace_id 找到失败原因。")
    print()

    telemetry3 = Telemetry("demo_scene3.jsonl")

    # 模拟一次完整的交互（包含成功和失败）
    with trace(telemetry3) as tid_success:
        print(f"🔖 追踪 A (正常)：{tid_success}")
        telemetry3.log_llm_call(100, 200, 800, success=True)
        telemetry3.log_tool_request("calculator", {"a": 10, "b": 5, "operation": "add"})
        telemetry3.log_tool_execution("calculator", 15, duration_ms=2, success=True)
        print("   ✅ 全部成功")

    print()

    with trace(telemetry3) as tid_fail:
        print(f"🔖 追踪 B (有故障)：{tid_fail}")
        telemetry3.log_llm_call(200, 0, 3000, success=False,
                                error="JSON 解析失败：输出不是有效 JSON")
        telemetry3.log_tool_request("calculator", {"a": 1, "b": 0, "operation": "divide"},
                                    duration_ms=30)
        telemetry3.log_tool_execution("calculator", None, duration_ms=1, success=False,
                                      error="division by zero")
        print("   ❌ 有失败事件")

    # 调试追踪 B
    print(f"\n🔍 调试追踪 B ({tid_fail})：")
    telemetry3.print_trace_detail(tid_fail)

    # 打印摘要
    telemetry3.print_summary()

    telemetry3.clear_log()

    # ================================================================
    # 场景 4：指标监控 —— 运行多轮操作后看聚合数据
    # ================================================================

    print("\n\n" + "=" * 60)
    print("  场景 4：指标监控——运行多轮操作观察指标变化")
    print("=" * 60)
    print()
    print("📝 运行 20 次结构化输出调用，观察成功率、平均延迟等指标。")
    print()

    agent4 = Agent(model="qwen2.5:7b", log_file="demo_scene4.jsonl")

    batch_size = 20
    print(f"正在运行 {batch_size} 次结构化输出调用...\n")

    for i in range(batch_size):
        result = agent4.generate_structured(
            f"生成一个包含数字 {i} 的 JSON",
            '{"number": "int", "squared": "int", "is_even": "bool"}'
        )
        if (i + 1) % 5 == 0:
            print(f"  进度：{i + 1}/{batch_size}")

    print(f"\n✅ {batch_size} 次调用完成\n")

    # 查看指标
    print("📊 运行指标：")
    m = agent4.telemetry.metrics
    print(f"   LLM 调用次数：{m.llm_calls}")
    print(f"   LLM 成功率：{m.llm_success_rate * 100:.2f}%")
    print(f"   LLM 失败次数：{m.llm_failures}")
    print(f"   平均延迟：{m.avg_latency_ms:.0f}ms")
    print(f"   总重试次数：{m.llm_retries}")
    print()
    agent4.show_telemetry()

    # 最终查看
    print("\n\n" + "=" * 60)
    print("  📊 最终 Agent 状态")
    print("=" * 60)
    print()
    agent4.show_info()

    agent4.telemetry.clear_log()

    print("\n\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


def interactive_mode():
    """交互模式：体验遥测系统"""

    print("=" * 60)
    print("  第十二课：遥测系统 - 交互模式")
    print("=" * 60)
    print()
    print("命令：")
    print("  agent    - 创建一个带遥测的 Agent 并测试")
    print("  tele     - 手动操作 Telemetry（模拟各种事件）")
    print("  trace    - 查看当前追踪详情")
    print("  metrics  - 查看指标摘要")
    print("  help     - 显示此帮助")
    print("  quit     - 退出")
    print()

    agent = None
    telemetry = Telemetry("agent_telemetry.jsonl")

    while True:
        try:
            user_input = input("🎯 命令: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "help":
            print()
            print("命令：")
            print("  agent    - 创建 Agent 并测试遥测")
            print("  tele     - 手动测试 Telemetry")
            print("  trace    - 查看追踪详情")
            print("  metrics  - 查看指标摘要")
            print("  help     - 显示此帮助")
            print("  quit     - 退出")
            print()
            continue

        if user_input.lower() == "agent":
            agent = Agent(model="qwen2.5:7b")
            print("\n✅ Agent 已创建（遥测已激活）\n")

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

            # 测试几个方法
            print("\n--- 测试结构化输出 ---")
            r1 = agent.generate_structured(
                "1+1 等于几？",
                '{"answer": "string", "value": "int"}'
            )
            print(f"   结果：{r1}")

            print("\n--- 测试工具调用 ---")
            tc = agent.request_tool("请计算 100 / 25")
            if tc:
                print(f"   工具请求：{tc}")
                r2 = agent.execute_tool_call(tc)
                print(f"   执行结果：{r2}")

            print("\n--- 测试对话 ---")
            r3 = agent.chat("你好！")
            print(f"   回复：{r3[:60]}...")

            # 查看遥测
            print()
            agent.show_telemetry()

        elif user_input.lower() == "tele":
            print("\n手动遥测演示：\n")

            tid = telemetry.start_trace()
            print(f"🔖 追踪 ID：{tid}")

            telemetry.log_llm_call(100, 200, 1500, success=True)
            telemetry.log_llm_call(50, 0, 3000, success=False, error="JSON parse failed")
            telemetry.log_tool_request("search", {"query": "Python"})
            telemetry.log_tool_execution("search", "结果列表...", duration_ms=200, success=True)
            telemetry.log_memory_op("add", "用户喜欢 Python")
            telemetry.log_decision(["chat", "tool", "exit"], "chat", duration_ms=400)

            print()
            telemetry.print_summary()

            print(f"\n🔍 追踪详情：")
            telemetry.print_trace_detail(tid)

        elif user_input.lower() == "trace":
            if agent and agent.telemetry and agent.telemetry.current_trace_id:
                print(f"\n🔍 当前追踪 ID：{agent.telemetry.current_trace_id}")
                agent.analyze_trace()
            else:
                print("\n⚠️ 没有活跃的 Agent 或追踪，请先输入 'agent' 创建。")

        elif user_input.lower() == "metrics":
            if agent:
                print()
                agent.show_telemetry()
            else:
                print("\n⚠️ 没有活跃的 Agent，请先输入 'agent' 创建。")

        else:
            print("❌ 未知命令。输入 'help' 查看可用命令。")

        print()


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_12_telemetry()


if __name__ == "__main__":
    main()
