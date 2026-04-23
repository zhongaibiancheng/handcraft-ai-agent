"""
第十课完整示例 - AoT（思维原子）系统

运行方式：
  python complete_example.py          # 演示模式
  python complete_example.py --demo   # 演示模式（同上）
  python complete_example.py --interact # 交互模式
"""

import sys
from agent.agent import Agent


def lesson_10_aot():
    """第十课演示：AoT（思维原子）系统"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第十课：AoT（思维原子）——让 Agent 学会排兵布阵")
    print("=" * 60)
    print()
    print("📝 说明：Agent 会将目标转换为带依赖关系的执行图。")
    print("   无依赖的节点可以并行执行，有依赖的节点按顺序执行。")
    print()

    # ========== 场景 1：基础 AoT 图生成 ==========

    print("=" * 60)
    print("  场景 1：生成 AoT 依赖图")
    print("=" * 60)
    print()

    goal1 = "研究并写一篇关于 AI Agent 的技术博客"
    print(f"🎯 目标：{goal1}\n")

    graph1 = agent.create_aot_plan(goal1)

    if graph1:
        agent.print_aot_graph(graph1)

        print("\n🚀 开始执行\n")
        results1 = agent.execute_aot_plan(graph1)
        print(f"📊 执行完成，共 {len(results1)} 个节点")
    else:
        print("❌ AoT 图生成失败")

    # ========== 场景 2：复杂任务（多依赖关系） ==========

    print("\n\n" + "=" * 60)
    print("  场景 2：复杂任务 —— 多依赖关系")
    print("=" * 60)
    print()

    goal2 = "搭建一个电商网站"
    print(f"🎯 目标：{goal2}\n")

    graph2 = agent.create_aot_plan(goal2)

    if graph2:
        agent.print_aot_graph(graph2)

        # 对比：如果顺序执行需要多少步
        total = len(graph2["nodes"])
        parallel = sum(1 for n in graph2["nodes"] if not n.get("depends_on"))
        max_depth = agent._calc_graph_depth(graph2)
        print(f"\n  📊 顺序执行需要 {total} 步")
        print(f"  📊 并行执行（理论最优）需要 {max_depth} 步")
        print(f"  📊 效率提升：{(1 - max_depth / total) * 100:.0f}%")

        print("\n🚀 开始执行\n")
        results2 = agent.execute_aot_plan(graph2)
        print(f"📊 执行完成，共 {len(results2)} 个节点")
    else:
        print("❌ AoT 图生成失败")

    # ========== 场景 3：从计划到 AoT 图（完整链路） ==========

    print("\n\n" + "=" * 60)
    print("  场景 3：目标 → AoT 图 → 执行（完整链路）")
    print("=" * 60)
    print()

    goal3 = "策划一场面向初中生的 AI 编程体验课"
    print(f"🎯 目标：{goal3}\n")

    # 生成 AoT 图（直接从目标）
    print("--- 生成 AoT 图 ---")
    graph3 = agent.create_aot_plan(goal3)

    if graph3:
        print("\n--- 执行 ---")
        results3 = agent.execute_aot_plan(graph3)

        print(f"\n📊 执行结果汇总：")
        for r in results3:
            status = r.get("status", "?")
            action = r.get("action", "?")
            node_id = r.get("id", "?")
            print(f"  ✅ [{node_id}] {action} → {status}")
    else:
        print("❌ AoT 图生成失败")

    # ========== 场景 4：循环依赖检测 ==========

    print("\n\n" + "=" * 60)
    print("  场景 4：循环依赖检测（安全机制演示）")
    print("=" * 60)
    print()
    print("📝 构造一个有循环依赖的图来演示安全机制：")
    print()

    bad_graph = {
        "nodes": [
            {"id": "1", "action": "步骤A（依赖步骤B）", "depends_on": ["2"]},
            {"id": "2", "action": "步骤B（依赖步骤A）", "depends_on": ["1"]},
        ]
    }

    print("  [1] 步骤A（依赖步骤B）")
    print("  [2] 步骤B（依赖步骤A）")
    print("  → 循环依赖！谁也执行不了\n")

    from agent.planner import execute_graph
    results_bad = execute_graph(bad_graph, lambda a: f"Executed: {a}")

    if len(results_bad) == 0:
        print("\n✅ 安全机制生效：循环依赖被检测到，图未执行")
    else:
        print(f"\n⚠️ 执行了 {len(results_bad)} 个节点（不应发生）")

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
    """交互模式：用户输入目标，Agent 生成 AoT 图并执行"""

    agent = Agent(model="qwen2.5:7b")

    print("=" * 60)
    print("  第十课：AoT（思维原子）系统 - 交互模式")
    print("=" * 60)
    print()
    print("💬 输入一个目标，Agent 会生成依赖图并按序执行。")
    print("   输入 'quit' 或 'exit' 退出。")
    print("   输入 'state' 查看当前状态。")
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

        print()

        # 生成 AoT 图
        graph = agent.create_aot_plan(user_input)

        if graph:
            agent.print_aot_graph(graph)

            # 询问是否执行
            try:
                confirm = input("\n▶️ 执行这个图吗？(y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if confirm == "y":
                print()
                results = agent.execute_aot_plan(graph)
                print(f"\n📊 执行完成，共 {len(results)} 个节点")
            else:
                print("已取消执行")
        else:
            print("❌ AoT 图生成失败，请重试。")

        print()


def main():
    if "--interact" in sys.argv:
        interactive_mode()
    else:
        lesson_10_aot()


if __name__ == "__main__":
    main()
