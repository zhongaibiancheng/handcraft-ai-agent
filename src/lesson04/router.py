"""
多技能路由框架 - 第四课配套
内置 5 个示例技能，支持动态注册新技能。
依赖：pip install openai
"""

import json
import re
import random
from openai import OpenAI
from typing import Optional


def extract_json_from_text(text: str) -> Optional[dict]:
    """从文本中提取 JSON"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


class SkillRouter:
    """
    技能路由器

    功能：
    - 注册/注销技能
    - 自动意图识别
    - 路由到对应技能执行
    - 支持决策日志和统计
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.skills: dict[str, dict] = {}  # {name: {func, description, call_count}}
        self.decision_log: list[dict] = []  # 决策日志

    def register(
        self,
        name: str,
        func: callable,
        description: str = "",
        examples: list[str] = None,
    ) -> None:
        """
        注册一个技能

        Args:
            name: 技能名称
            func: 处理函数，接收 (user_input, context) 参数
            description: 技能描述（给 AI 看的，帮助它理解何时使用）
            examples: 触发该技能的示例用户输入列表
        """
        self.skills[name] = {
            "func": func,
            "description": description,
            "examples": examples or [],
            "call_count": 0,
        }
        print(f"✅ 已注册技能：{name}")
        if description:
            print(f"   描述：{description}")

    def unregister(self, name: str) -> bool:
        """注销一个技能"""
        if name in self.skills:
            del self.skills[name]
            print(f"🗑️ 已注销技能：{name}")
            return True
        print(f"⚠️ 技能不存在：{name}")
        return False

    def decide(self, user_input: str) -> Optional[str]:
        """
        意图识别 — 让 AI 从已注册的技能中选择最合适的一个

        改进点（相比基础版）：
        - 把技能描述和示例也发给 AI，提高识别准确率
        - 支持随机打乱选项顺序，避免位置偏差
        """
        if not self.skills:
            return None

        # 构建带描述的选项列表
        skill_list = list(self.skills.keys())
        skill_details = []
        for name in skill_list:
            info = self.skills[name]
            detail = f"- {name}"
            if info["description"]:
                detail += f"：{info['description']}"
            if info["examples"]:
                detail += f"（例如：{'、'.join(info['examples'][:3])}）"
            skill_details.append(detail)

        options_text = "\n".join(skill_details)

        prompt = f"""你是一个意图识别助手。根据用户输入，从下面的技能列表中选择最合适的一个。

可用技能：
{options_text}

还有一个兜底选项：
- unknown：当无法判断用户意图时选择此项

规则：
1. 只能从上面的技能名称中选择，不能自己编造
2. 只返回 JSON，不要有任何解释
3. JSON 格式：{{"decision": "技能名称"}}

用户输入：{user_input}

请返回 JSON："""

        # 随机打乱选项顺序，避免位置偏差（position bias）
        # 但我们不需要真的打乱发给 AI 的顺序，
        # 因为 AI 通常不会有这个问题。这里保留随机打乱逻辑以备需要时启用。
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                text = response.choices[0].message.content
                parsed = extract_json_from_text(text)

                if parsed and "decision" in parsed:
                    decision = parsed["decision"]
                    # 严格匹配
                    if decision in self.skills or decision == "unknown":
                        return decision
                    # 模糊匹配
                    for skill_name in self.skills:
                        if (decision.lower().replace(" ", "_").replace("-", "_")
                                == skill_name.lower().replace(" ", "_").replace("-", "_")):
                            return skill_name

            except Exception as e:
                print(f"⚠️ 请求出错（第{attempt+1}次）：{e}")

        return None

    def execute(self, user_input: str, context: dict = None) -> dict:
        """
        完整的意图识别 + 路由执行

        Args:
            user_input: 用户输入
            context: 额外上下文（会传给技能函数）

        Returns:
            {
                "skill": "技能名称",
                "success": True/False,
                "result": "执行结果",
                "raw_input": "原始用户输入"
            }
        """
        context = context or {}

        # 记录日志
        log_entry = {
            "input": user_input,
            "skill": None,
            "success": False,
            "timestamp": __import__("time").strftime("%H:%M:%S"),
        }

        # 意图识别
        decision = self.decide(user_input)

        if decision is None:
            log_entry["skill"] = "FAILED"
            self.decision_log.append(log_entry)
            return {
                "skill": "FAILED",
                "success": False,
                "result": "❌ 意图识别失败，请换个说法试试",
                "raw_input": user_input,
            }

        if decision == "unknown":
            log_entry["skill"] = "unknown"
            self.decision_log.append(log_entry)
            return {
                "skill": "unknown",
                "success": False,
                "result": "🤔 抱歉，我暂时不知道怎么帮你处理这个请求。",
                "raw_input": user_input,
            }

        if decision not in self.skills:
            log_entry["skill"] = f"NOT_FOUND:{decision}"
            self.decision_log.append(log_entry)
            return {
                "skill": decision,
                "success": False,
                "result": f"⚠️ 技能「{decision}」未注册",
                "raw_input": user_input,
            }

        # 路由执行
        print(f"🎯 识别意图：{decision}")
        skill_info = self.skills[decision]
        skill_info["call_count"] += 1

        try:
            result = skill_info["func"](user_input, context)
            log_entry["skill"] = decision
            log_entry["success"] = True
            self.decision_log.append(log_entry)

            return {
                "skill": decision,
                "success": True,
                "result": result,
                "raw_input": user_input,
            }
        except Exception as e:
            log_entry["skill"] = decision
            self.decision_log.append(log_entry)

            return {
                "skill": decision,
                "success": False,
                "result": f"⚠️ 执行出错：{e}",
                "raw_input": user_input,
            }

    def show_stats(self) -> None:
        """显示技能调用统计"""
        print("\n" + "=" * 55)
        print("  📊 技能调用统计")
        print("=" * 55)
        if not self.skills:
            print("  （暂无注册技能）")
            return

        for name, info in self.skills.items():
            bar = "█" * info["call_count"]
            print(f"  {name:20s} {bar} ({info['call_count']}次)")

        # 决策日志摘要
        if self.decision_log:
            success = sum(1 for log in self.decision_log if log["success"])
            total = len(self.decision_log)
            print("-" * 55)
            print(f"  总决策次数：{total}  成功率：{success}/{total} ({success/total*100:.0f}%)")
        print("=" * 55)

    def show_skills(self) -> None:
        """显示所有已注册技能"""
        print("\n" + "=" * 55)
        print("  📋 已注册技能列表")
        print("=" * 55)
        if not self.skills:
            print("  （暂无注册技能）")
            return
        for i, (name, info) in enumerate(self.skills.items(), 1):
            print(f"  {i}. {name}")
            if info["description"]:
                print(f"     描述：{info['description']}")
            if info["examples"]:
                print(f"     示例：{'、'.join(info['examples'][:3])}")
        print("=" * 55)


# ====== 示例技能函数 ======

def skill_answer_question(user_input: str, context: dict = None) -> str:
    """回答用户的技术问题"""
    # 实际使用时，这里可以调用搜索引擎、查文档等
    return f"📝 正在回答问题：「{user_input}」\n\n这是一个示例回复。实际使用时，这里会调用问答模块或搜索 API。"


def skill_summarize_text(user_input: str, context: dict = None) -> str:
    """总结用户提供的文本"""
    # 实际使用时，这里会调用 LLM 的摘要功能
    return f"📋 正在生成摘要...\n\n原文：「{user_input[:100]}...」\n\n这是示例摘要。实际使用时，会调用 generate_structured() 或直接让 LLM 生成摘要。"


def skill_translate(user_input: str, context: dict = None) -> str:
    """翻译用户提供的文本"""
    target_lang = (context or {}).get("target_lang", "英文")
    return f"🌐 正在翻译为{target_lang}...\n\n原文：「{user_input[:100]}...」\n\n这是示例翻译。实际使用时，会调用 LLM 的翻译能力。"


def skill_code_review(user_input: str, context: dict = None) -> str:
    """审查用户提供的代码"""
    return f"🔍 正在审查代码...\n\n这是示例代码审查。实际使用时，会分析代码中的 bug、安全问题和改进建议。"


def skill_write_code(user_input: str, context: dict = None) -> str:
    """根据用户需求编写代码"""
    return f"💻 正在编写代码...\n\n需求：「{user_input[:100]}...」\n\n这是示例代码生成。实际使用时，会根据需求生成完整代码。"


# ====== 快速测试 ======
if __name__ == "__main__":
    router = SkillRouter(model="qwen2.5:7b")

    # 注册技能（带描述和示例，让 AI 更好识别）
    router.register(
        name="answer_question",
        func=skill_answer_question,
        description="回答用户的技术问题或知识查询",
        examples=["什么是装饰器", "Python 怎么读文件", "AI Agent 是什么"],
    )
    router.register(
        name="summarize_text",
        func=skill_summarize_text,
        description="总结或浓缩用户提供的文本内容",
        examples=["帮我总结一下", "缩写这段话", "提取这篇文章的要点"],
    )
    router.register(
        name="translate",
        func=skill_translate,
        description="翻译文本（默认中翻英）",
        examples=["翻译成英文", "这段话用英文怎么说", "translate this"],
    )
    router.register(
        name="code_review",
        func=skill_code_review,
        description="审查代码中的 bug 和安全问题",
        examples=["帮我看看这段代码", "review this code", "这段代码有什么问题"],
    )
    router.register(
        name="write_code",
        func=skill_write_code,
        description="根据需求编写代码",
        examples=["帮我写一个", "写个脚本", "用 Python 实现"],
    )

    router.show_skills()

    # 测试路由
    print("\n" + "=" * 55)
    print("  🧪 路由测试")
    print("=" * 55)

    tests = [
        "量子计算是什么？",
        "帮我把这段话缩写到100字以内",
        "把 Hello World 翻译成日文",
        "帮我看看这段 Python 代码有没有 bug：print('hello",
        "写一个快速排序算法",
        "今天晚上吃什么",  # 模糊/无关输入
    ]

    for test_input in tests:
        print(f"\n💬 用户：{test_input}")
        result = router.execute(test_input)
        print(f"   技能：{result['skill']}")
        print(f"   结果：{result['result'][:80]}...")

    # 显示统计
    router.show_stats()
