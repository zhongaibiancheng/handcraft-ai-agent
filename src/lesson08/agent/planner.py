"""
规划器模块 - 第八课引入

职责：根据目标生成步骤计划。
和 Agent 类分离——规划逻辑独立，方便后续替换和扩展。

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

    示例返回值：
        {"steps": ["研究目标领域", "制定方案", "执行方案", "验证结果"]}
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
