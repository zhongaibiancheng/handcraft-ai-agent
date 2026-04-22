"""
规划器模块 - 第八课引入，第九课升级

职责：
  - create_plan()：根据目标生成步骤计划（第八课）
  - create_atomic_action()：将步骤转换为原子动作（第九课新增）

依赖：shared.utils.extract_json_from_text（第三课引入）
"""

from typing import Optional


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

    示例：
        step = "撰写初稿"
        → {"action": "generate_text", "inputs": {"topic": "...", "length": "..."}}
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
            # 确保 inputs 字段存在且为字典
            if "inputs" not in action or not isinstance(action["inputs"], dict):
                action["inputs"] = {}
            return action

    return None
