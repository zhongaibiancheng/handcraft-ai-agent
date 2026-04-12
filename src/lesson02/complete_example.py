"""
完整命令行交互程序 - 第二课：System Prompt 角色管理
功能：
  - 可视化角色选择菜单
  - 实时切换 AI 角色
  - 流式输出（打字机效果）
  - 多轮对话支持
  - 对话历史查看/清空

运行方式：
  python complete_example.py

依赖：
  pip install openai
  需要先启动 Ollama 服务（ollama serve）并下载 qwen2.5:7b 模型
"""

import sys
import time
from agent import Agent
from roles import ROLES, list_roles


def print_banner():
    """打印欢迎界面"""
    banner = """
╔══════════════════════════════════════════╗
║                                          ║
║   🤖 AI Agent 角色对话系统 v1.0          ║
║   手搓 AI Agent 从 0 到 1 — 第二课       ║
║                                          ║
║   公众号「开源情报局」回复 Agent 获取    ║
║   本项目完整源码                          ║
║                                          ║
╚══════════════════════════════════════════╝
"""
    print(banner)


def show_role_menu():
    """显示角色选择菜单"""
    print("\n📋 可选角色列表：\n")
    
    # 将角色转为有序列表
    role_keys = list(ROLES.keys())
    for i, key in enumerate(role_keys, 1):
        info = ROLES[key]
        tags = " | ".join(info["tags"])
        print(f"  {i:2d}. {info['name']}")
        print(f"      关键字: {key}  |  标签: {tags}")
    
    print(f"\n  {len(role_keys) + 1:2d}. 🆕 自定义角色（自己输入提示词）")
    print(f"  {len(role_keys) + 2:2d}. ℹ️  查看当前配置")
    print(f"  {len(role_keys) + 3:2d}. 🗑️  清空对话历史")
    print(f"  {len(role_keys) + 4:2d}. ❌ 退出程序")
    print()
    
    return role_keys


def select_role(agent: Agent, role_keys: list[str]) -> str:
    """
    让用户选择角色，返回选中的 System Prompt
    
    Args:
        agent: Agent 实例
        role_keys: 所有可用角色的关键字列表
        
    Returns:
        选中的 System Prompt 字符串
    """
    while True:
        try:
            choice = input("请选择角色编号 > ").strip()
            idx = int(choice)
            
            if 1 <= idx <= len(role_keys):
                key = role_keys[idx - 1]
                prompt = ROLES[key]["prompt"]
                name = ROLES[key]["name"]
                agent.set_role(prompt)
                print(f"\n✅ 已切换为角色：{name}\n")
                return prompt
            
            elif idx == len(role_keys) + 1:
                # 自定义角色
                custom = input("请输入你的 System Prompt > ").strip()
                if custom:
                    agent.set_role(custom)
                    print(f"\n✅ 已设置为自定义角色\n")
                    return custom
                else:
                    print("⚠️ 输入不能为空\n")
                    
            elif idx == len(role_keys) + 2:
                agent.show_info()
                
            elif idx == len(role_keys) + 3:
                agent.clear_history()
                
            elif idx == len(role_keys) + 4:
                print("\n👋 再见！")
                sys.exit(0)
                
            else:
                print(f"\n⚠️ 请输入 1-{len(role_keys) + 4} 之间的数字\n")
                
        except ValueError:
            print("\n⚠️ 请输入有效的数字\n")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)


def chat_loop(agent: Agent, system_prompt: str):
    """
    主聊天循环
    
    Args:
        agent: Agent 实例
        system_prompt: 当前使用的 System Prompt
    """
    print("💬 开始对话（输入 /role 切换角色，输入 /clear 清空历史，输入 /quit 退出）\n")
    
    while True:
        try:
            user_input = input("👤 你 > ").strip()
            
            if not user_input:
                continue
            
            # 命令处理
            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/role":
                role_keys = list(ROLES.keys())
                system_prompt = select_role(agent, role_keys)
                print("💬 已切换角色，继续输入问题吧~\n")
                continue
            elif user_input.lower() == "/clear":
                agent.clear_history()
                continue
            
            # 流式输出回复
            print("🤖 AI > ", end="", flush=True)
            
            for token in agent.chat_stream(user_input):
                print(token, end="", flush=True)
            
            print("\n")  # 回车换行
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}\n")
            print("提示：确认 Ollama 服务是否在运行（ollama serve）\n")


def lesson_02_demo():
    """
    第二课演示：同一问题，三种不同角色的效果对比
    不需要用户交互，直接展示效果
    """
    print("=" * 60)
    print("  🎬 第二课演示：System Prompt 的魔力")
    print("  同一个问题，三种角色，三种完全不同的回答")
    print("=" * 60)
    print()
    
    agent = Agent(model="qwen2.5:7b", temperature=0.7)
    question = "什么是 AI Agent？"
    
    # 三种角色
    demos = [
        ("python_teacher", "🎓 老师模式"),
        ("eli5_explainer", "🗣 大白话模式"),
        ("concise_responder", "⚡ 极简模式"),
    ]
    
    for role_key, label in demos:
        agent.set_role(getattr(__import__('roles'), 'get_role')(role_key))
        
        print(f"{label}")
        print("-" * 40)
        reply = agent.generate_with_role(question)
        print(reply)
        print()
    
    print("=" * 60)
    print("  ✅ 演示完成！相同的问题，不同的角色设定")
    print("     得到了风格迥异的回复。这就是 System Prompt！")
    print("=" * 60)


def main():
    """主入口"""
    print_banner()
    
    # 先检查参数
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # 演示模式：自动运行对比测试
        lesson_02_demo()
        return
    
    # 初始化 Agent
    agent = Agent(model="qwen2.5:7b")
    
    print("💡 提示：请先选择一个角色，然后开始对话\n")
    
    # 角色选择
    role_keys = show_role_menu()
    system_prompt = select_role(agent, role_keys)
    
    # 进入聊天循环
    chat_loop(agent, system_prompt)
    
    print("\n👋 感谢使用，下次见！")


# ====== 入口 ======
if __name__ == "__main__":
    main()
