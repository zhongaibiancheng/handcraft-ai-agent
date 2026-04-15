"""
完整 CLI 交互程序 - 第四课升级版
支持：连续对话 + 自动意图识别 + 技能切换 + 演示模式
依赖：pip install openai
"""

import sys
import time
from agent import Agent


def print_slow(text: str, delay: float = 0.02) -> None:
    """模拟打字机效果输出文本"""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def print_separator() -> None:
    """打印分隔线"""
    print("\n" + "─" * 55)


def demo_mode(agent: Agent) -> None:
    """演示模式：自动展示多种决策场景"""
    print("\n🎭 ====== 演示模式 ======\n")
    print("接下来会自动测试6个不同的用户输入，展示 AI 如何自动识别意图并路由。\n")

    # 注册示例技能
    def skill_answer(text, ctx=None):
        return f"📝 [回答问题] 收到问题：「{text}」，正在查找答案..."

    def skill_summarize(text, ctx=None):
        return f"📋 [文本摘要] 正在为「{text[:30]}...」生成摘要..."

    def skill_translate(text, ctx=None):
        return f"🌐 [翻译] 正在翻译：「{text[:30]}...」"

    agent.register_skill("answer_question", skill_answer, "回答问题")
    agent.register_skill("summarize_text", skill_summarize, "摘要总结")
    agent.register_skill("translate", skill_translate, "翻译")

    test_cases = [
        ("量子计算是什么？", "❓ 知识问答"),
        ("帮我总结一下这篇文章的主要内容", "📋 文本摘要"),
        ("把这段话翻译成日文", "🌐 翻译任务"),
        ("Python 的装饰器怎么用？", "❓ 编程问题"),
        ("请帮我把这个会议记录缩写到200字", "📋 会议摘要"),
        ("嗯...好的...", "🤔 模糊输入"),
    ]

    for i, (user_input, label) in enumerate(test_cases, 1):
        print_separator()
        print(f"测试 {i}/6：{label}")
        print(f"💬 用户输入：{user_input}")
        print()

        start = time.time()
        result = agent.route(user_input)
        elapsed = time.time() - start

        print(f"⏱ 耗时：{elapsed:.1f}秒")
        print(f"📤 路由结果：{result}")
        time.sleep(1)

    print_separator()
    print("\n✅ 演示完成！")
    print("\n💡 你看到了：同样的代码，不同的输入，AI 自动选择了不同的技能来处理。")
    print("   这就是「决策」—— Agent 的核心能力。\n")


def interactive_mode(agent: Agent) -> None:
    """交互模式：连续对话 + 自动路由"""
    print("\n💬 ====== 交互模式 ======\n")

    # 注册默认技能
    def skill_answer(text, ctx=None):
        # 实际使用时，这里可以调用搜索引擎或知识库
        reply = agent.generate_with_role(f"请回答以下问题：{text}")
        return f"📝 {reply}"

    def skill_summarize(text, ctx=None):
        reply = agent.generate_with_role(f"请总结以下内容，控制在100字以内：{text}")
        return f"📋 摘要：{reply}"

    def skill_translate(text, ctx=None):
        reply = agent.generate_with_role(f"请将以下内容翻译成英文：{text}")
        return f"🌐 翻译：{reply}"

    agent.register_skill("answer_question", skill_answer, "回答问题")
    agent.register_skill("summarize_text", skill_summarize, "摘要总结")
    agent.register_skill("translate", skill_translate, "翻译")

    agent.show_skills()

    print("\n使用说明：")
    print("  - 直接输入你想说的话，AI 会自动识别意图并路由")
    print("  - 输入「技能」查看已注册的技能")
    print("  - 输入「统计」查看决策统计")
    print("  - 输入「清空」清空对话历史")
    print("  - 输入「退出」结束程序")
    print_separator()

    while True:
        try:
            user_input = input("\n你：").strip()

            if not user_input:
                continue

            # 命令处理
            if user_input in ("退出", "exit", "quit", "q"):
                print("\n👋 再见！\n")
                break

            elif user_input == "技能":
                agent.show_skills()
                continue

            elif user_input == "统计":
                if hasattr(agent, 'show_skills'):
                    agent.show_info()
                continue

            elif user_input == "清空":
                agent.clear_history()
                continue

            # 路由处理
            print_slow("🤔 思考中...")
            result = agent.route(user_input)
            print(f"\n🤖 {result}")

        except KeyboardInterrupt:
            print("\n\n👋 再见！\n")
            break
        except Exception as e:
            print(f"\n⚠️ 出错了：{e}")


def main():
    """主入口"""
    print("""
╔══════════════════════════════════════════════╗
║   🤖 AI Agent 决策系统 — 第四课演示         ║
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

    # 选择模式
    if "--demo" in sys.argv:
        demo_mode(agent)
    else:
        print("选择模式：")
        print("  1. 🎭 演示模式（自动展示决策效果）")
        print("  2. 💬 交互模式（手动输入测试）")
        print()

        choice = input("请选择（1/2，默认2）：").strip()

        if choice == "1":
            demo_mode(agent)
        else:
            interactive_mode(agent)


if __name__ == "__main__":
    main()
