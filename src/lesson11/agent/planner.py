"""
规划器模块 - 第八课引入，第九课升级，第十课再升级

职责：
  - create_plan()：根据目标生成步骤计划（第八课）
  - create_atomic_action()：将步骤转换为原子动作（第九课新增）
  - create_aot_graph()：生成 AoT 依赖图（第十课新增）
  - execute_graph()：按依赖关系执行图（第十课新增）

依赖：shared.utils.extract_json_from_text（第三课引入）
"""

from typing import Optional, Callable


def create_plan(llm, goal: str) -> Optional[dict]:
    """
    生成实现目标的步骤计划。

    用于：第八课

    Args:
        llm: 语言模型实例（需要有 generate 方法）
        goal: 要实现的目标

    Returns:
        包含 "steps" 列表的字典，失败则返回 None
    """
    from shared.utils import extract_json_from_text

    prompt = f"""为以下目标制定一个详细的执行计划。只返回有效的 JSON。

规则：
1. 只返回有效的 JSON
2. 不要任何解释，不要 Markdown
3. 直接以 {{ 开头，以 }} 结尾
4. 步骤应该具体、可执行、有逻辑顺序
5. 步骤数量根据任务复杂度决定，一般 3～7 步

JSON 格式：
{{"steps": ["步骤1的描述", "步骤2的描述", "步骤3的描述"]}}

目标：{goal}

请返回 JSON："""

    for attempt in range(3):
        response = llm.generate(prompt, temperature=0.0)
        plan = extract_json_from_text(response)

        if plan and "steps" in plan and isinstance(plan["steps"], list):
            if len(plan["steps"]) > 0:
                return plan

    return None


def create_atomic_action(llm, step: str) -> Optional[dict]:
    """
    将计划步骤转换为原子动作。

    用于：第九课

    Args:
        llm: 语言模型实例（需要有 generate 方法）
        step: 计划中的一个步骤

    Returns:
        原子动作字典（带 "action" 和 "inputs"），失败则返回 None
    """
    from shared.utils import extract_json_from_text

    prompt = f"""将以下计划步骤转换为一个具体的原子动作。只返回有效的 JSON。

规则：
1. 只返回有效的 JSON
2. 不要任何解释，不要 Markdown
3. 直接以 {{ 开头，以 }} 结尾
4. action 应该是一个简单、具体的操作名称（如 generate_text、search_web、send_email）
5. inputs 应该包含执行该动作所需的全部参数
6. 参数值必须具体，不要模糊

JSON 格式：
{{"action": "动作名称", "inputs": {{"参数名": "具体参数值"}}}}

步骤：{step}

请返回 JSON："""

    for attempt in range(3):
        response = llm.generate(prompt, temperature=0.0)
        action = extract_json_from_text(response)

        if action and "action" in action:
            if "inputs" not in action or not isinstance(action["inputs"], dict):
                action["inputs"] = {}
            return action

    return None


def create_aot_graph(llm, goal: str) -> Optional[dict]:
    """
    生成 AoT 执行图。

    用于：第十课

    Args:
        llm: 语言模型实例（需要有 generate 方法）
        goal: 要实现的目标

    Returns:
        带 nodes 的 AoT 图，失败返回 None

    示例：
        goal = "研究并写一篇博客"
        → {
            "nodes": [
                {"id": "1", "action": "研究主题", "depends_on": []},
                {"id": "2", "action": "确定大纲", "depends_on": []},
                {"id": "3", "action": "撰写初稿", "depends_on": ["1", "2"]},
                {"id": "4", "action": "审校修改", "depends_on": ["3"]}
            ]
        }
    """
    from shared.utils import extract_json_from_text

    prompt = f"""为以下目标创建一个执行图。只返回有效的 JSON。

规则：
1. 只返回有效的 JSON
2. 不要任何解释，不要 Markdown
3. 直接以 {{ 开头，以 }} 结尾
4. 每个节点必须有 id、action、depends_on
5. id 使用数字字符串（"1", "2", "3"...）
6. depends_on 列表中的 id 必须是其他节点的 id
7. 不存在循环依赖
8. 没有依赖的节点表示可以并行执行
9. 步骤数量根据任务复杂度决定，一般 4～8 个节点

JSON 格式：
{{"nodes": [{{"id": "1", "action": "具体动作描述", "depends_on": []}}, {{"id": "2", "action": "具体动作描述", "depends_on": ["1"]}}]}}

目标：{goal}

请返回 JSON："""

    for attempt in range(3):
        response = llm.generate(prompt, temperature=0.0)
        graph = extract_json_from_text(response)

        if graph and "nodes" in graph and isinstance(graph["nodes"], list):
            if len(graph["nodes"]) == 0:
                continue

            # 第一层验证：节点结构完整性
            node_ids = set()
            for node in graph["nodes"]:
                if "id" not in node or "action" not in node or "depends_on" not in node:
                    break
                node_ids.add(node["id"])
            else:
                # 第二层验证：依赖引用是否指向有效节点
                all_deps_valid = True
                for node in graph["nodes"]:
                    for dep in node.get("depends_on", []):
                        if dep not in node_ids:
                            all_deps_valid = False
                            break
                    if not all_deps_valid:
                        break

                if all_deps_valid:
                    return graph

    return None


def execute_graph(graph: dict, action_fn: Callable[[str], str]) -> list[dict]:
    """
    按依赖关系执行图（拓扑排序）。

    用于：第十课

    Args:
        graph: AoT 图（带 "nodes" 列表）
        action_fn: 执行单个动作的函数，接收 action 字符串，返回结果

    Returns:
        执行结果列表
    """
    nodes = graph["nodes"]
    completed = set()
    results = []
    round_num = 0

    print(f"  📊 图中共 {len(nodes)} 个节点\n")

    while len(completed) < len(nodes):
        round_num += 1

        # 找出所有依赖已满足的节点
        ready = [
            n for n in nodes
            if n["id"] not in completed
            and all(d in completed for d in n.get("depends_on", []))
        ]

        if not ready:
            # 没有可执行的节点 → 循环依赖
            remaining = [n["id"] for n in nodes if n["id"] not in completed]
            print(f"  ⚠️ 检测到循环依赖，未完成节点：{remaining}")
            break

        print(f"  --- 第 {round_num} 轮执行（{len(ready)} 个就绪） ---")

        for node in ready:
            action = node["action"]
            deps = node.get("depends_on", [])
            dep_str = f"（依赖：{', '.join(deps)}）" if deps else "（无依赖）"

            result = action_fn(action)
            results.append({
                "id": node["id"],
                "action": action,
                "result": result,
                "status": "completed"
            })
            completed.add(node["id"])

            print(f"  ✅ [{node['id']}] {action} {dep_str}")

        print()

    return results
