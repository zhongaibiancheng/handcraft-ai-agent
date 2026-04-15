"""
Agent 类封装 - 第四课升级版：决策与路由
在第二课（System Prompt）+ 第三课（结构化输出）基础上，
新增 decide() 方法和 route() 路由系统。
依赖：pip install openai
"""

import json
import re
from openai import OpenAI
from typing import Optional, Generator


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    从文本中提取 JSON。

    因为模型经常会在 JSON 前后加一些废话，
    这个函数负责把真正的 JSON 找出来。

    尝试顺序：
    1. 直接把整个文本当 JSON 解析
    2. 用正则找第一个 { ... } 块
    """
    # 尝试1：直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试2：正则提取 { ... } 块
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


class Agent:
    """
    AI Agent 类 - 第四课升级版

    新增功能：
    - decide()：让 AI 从选项列表中做决策
    - route()：根据决策自动路由到对应技能
    - register_skill()：注册新技能（函数）
    - 内置 5 个示例技能

    保留功能（来自第二课）：
    - System Prompt 角色管理
    - 多轮对话 + 流式输出
    - generate_structured()（来自第三课）
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or "你是一个有帮助的AI助手。"
        self.temperature = temperature
        self.history: list[dict] = []

        # 第四课新增：技能注册表
        # 键 = 技能名称（决策选项），值 = {func, description, examples}
        self.skills: dict[str, dict] = {}

    # ==================== 第二课方法 ====================

    def set_role(self, system_prompt: str) -> None:
        """切换 Agent 角色，同时清空对话历史"""
        self.system_prompt = system_prompt
        self.history.clear()
        print(f"✅ 角色已切换：{system_prompt[:30]}...")

    def generate_with_role(self, user_input: str) -> str:
        """使用 System Prompt 生成回复（单轮，无历史）"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=self.temperature,
            stream=False,
        )
        return response.choices[0].message.content.strip()

    def chat(self, user_input: str) -> str:
        """带历史记忆的多轮对话"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=False,
        )

        reply = response.choices[0].message.content.strip()
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        if len(self.history) > 40:
            self.history = self.history[-40:]

        return reply

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """流式输出 — 打字机效果"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta
                yield delta

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_reply})

    # ==================== 第三课方法 ====================

    def generate_structured(self, user_input: str, schema: str) -> Optional[dict]:
        """
        强制输出结构化 JSON（第三课核心方法）

        Args:
            user_input: 用户的问题或请求
            schema: JSON Schema 描述

        Returns:
            解析后的字典，如果3次重试都失败则返回 None
        """
        prompt = f"""{self.system_prompt}

你必须严格按照以下 JSON 格式回复，不要添加任何其他内容。
只返回有效的 JSON，不要有解释、不要有 markdown 格式标记。

要求的 JSON 格式：
{schema}

用户请求：{user_input}

请返回 JSON："""

        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            text = response.choices[0].message.content
            parsed = extract_json_from_text(text)

            if parsed is not None:
                return parsed

        return None

    # ==================== 第四课新增方法 ====================

    def decide(self, user_input: str, choices: list[str]) -> Optional[str]:
        """
        让 AI 从选项列表中选择一个动作（第四课核心方法）

        AI 分析用户输入，从 choices 列表中选出最匹配的选项。
        不是生成自由文本，而是从预定义选项中选择。

        ⚠️ 本方法是基础版，只接收名称列表。
        如果技能有描述信息，请使用 route() 方法，它会自动带上描述。

        Args:
            user_input: 用户说的话
            choices: AI 可以选择的动作列表（名称）

        Returns:
            选中的动作名称，如果3次重试都失败则返回 None
        """
        options = "\n".join(f"- {choice}" for choice in choices)

        # 构造 User Prompt（技能说明放在 User Prompt 里，不动 System Prompt）
        user_prompt = f"""你是一个意图识别助手。根据用户输入，从下面的可选动作中选择最合适的一个。

可选动作：
{options}

规则：
1. 只能从上面的可选动作名称中选择一个，不能自己创造新名称
2. 只返回 JSON，不要有任何解释
3. JSON 格式：{{"decision": "动作名称"}}

用户输入：{user_input}

请返回 JSON："""

        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},   # 技能列表在 user prompt 里
                ],
                temperature=0.0,
            )

            text = response.choices[0].message.content
            parsed = extract_json_from_text(text)

            if parsed and "decision" in parsed:
                decision = parsed["decision"]
                # 关键：验证决策是否在选项列表中
                if decision in choices:
                    return decision
                # 尝试模糊匹配
                for choice in choices:
                    if decision.lower().replace("_", "").replace("-", "") == choice.lower().replace("_", "").replace("-", ""):
                        return choice

        return None

    def decide_with_descriptions(self, user_input: str) -> Optional[str]:
        """
        带描述信息的决策（推荐方法）

        和 decide() 的区别：
        - decide()：只传技能名称，AI 不知道每个技能是干嘛的
        - decide_with_descriptions()：把技能名称 + 描述 + 示例一起发给 AI，识别准确率高很多

        技能说明放在 User Prompt 里（不是 System Prompt），原因：
        - System Prompt 是角色设定，应该保持稳定
        - User Prompt 是每次请求的上下文，技能列表是动态的，放这里更合理

        Returns:
            选中的技能名称，如果失败则返回 None
        """
        if not self.skills:
            return None

        # 构建带描述的技能选项列表
        skill_options = []
        for name, info in self.skills.items():
            line = f"- {name}"
            if info.get("description"):
                line += f"：{info['description']}"
            if info.get("examples"):
                examples = "、".join(info["examples"][:3])
                line += f"（适用场景：{examples}）"
            skill_options.append(line)

        options_text = "\n".join(skill_options)

        user_prompt = f"""你是一个意图识别助手。根据用户输入，从下面的可用技能中选择最合适的一个。

可用技能：
{options_text}

兜底选项：
- unknown：当无法判断用户意图时选择此项

规则：
1. 只能从上面的技能名称中选择，不能自己创造新技能名
2. 只返回 JSON，不要有任何解释
3. JSON 格式：{{"decision": "技能名称"}}

用户输入：{user_input}

请返回 JSON："""

        choices = list(self.skills.keys()) + ["unknown"]

        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

            text = response.choices[0].message.content
            parsed = extract_json_from_text(text)

            if parsed and "decision" in parsed:
                decision = parsed["decision"]
                if decision in choices:
                    return decision
                # 模糊匹配
                for choice in choices:
                    if decision.lower().replace("_", "").replace("-", "") == choice.lower().replace("_", "").replace("-", ""):
                        return choice

        return None

    def register_skill(
        self,
        name: str,
        func: callable,
        description: str = "",
        examples: list[str] = None,
    ) -> None:
        """
        注册一个新技能

        Args:
            name: 技能名称（作为决策选项）
            func: 技能处理函数，接收 user_input 参数，返回处理结果
            description: 技能描述（给 AI 看的，帮助它理解何时使用这个技能）
            examples: 触发该技能的示例用户输入（给 AI 看的，提高识别准确率）
        """
        self.skills[name] = {
            "func": func,
            "description": description,
            "examples": examples or [],
        }
        print(f"✅ 已注册技能：{name}")
        if description:
            print(f"   描述：{description}")

    def route(self, user_input: str) -> str:
        """
        完整的意图识别 + 路由执行流程（推荐方法）

        步骤：
        1. 用 decide_with_descriptions() 让 AI 选择（带上技能描述和示例）
        2. 验证并路由到对应的技能函数
        3. 执行并返回结果

        Args:
            user_input: 用户输入

        Returns:
            技能执行结果，或错误提示
        """
        if not self.skills:
            return "❌ 没有注册任何技能。请先调用 register_skill() 注册技能。"

        # 使用带描述的决策方法
        decision = self.decide_with_descriptions(user_input)

        if decision is None:
            return "❌ 无法理解你的意图，请换个说法试试。"

        if decision == "unknown":
            return "🤔 抱歉，我暂时不知道怎么帮你处理这个请求。"

        # 路由到对应技能
        print(f"🎯 识别意图：{decision}")

        if decision in self.skills:
            try:
                return self.skills[decision]["func"](user_input)
            except Exception as e:
                return f"⚠️ 技能执行出错：{e}"

        return f"⚠️ 未知动作：{decision}"

    def show_skills(self) -> None:
        """打印所有已注册的技能"""
        if not self.skills:
            print("📋 当前没有注册任何技能")
            return

        print("=" * 55)
        print("  已注册技能列表：")
        print("-" * 55)
        for i, (name, info) in enumerate(self.skills.items(), 1):
            print(f"  {i}. {name}")
            if info.get("description"):
                print(f"     描述：{info['description']}")
            if info.get("examples"):
                print(f"     示例：{'、'.join(info['examples'][:3])}")
        print("=" * 55)

    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()
        print("🗑️ 对话历史已清空")

    def show_info(self) -> None:
        """打印当前 Agent 配置信息"""
        print("=" * 50)
        print(f"  模型：{self.model}")
        print(f"  角色：{self.system_prompt}")
        print(f"  温度：{self.temperature}")
        print(f"  历史轮数：{len(self.history) // 2}")
        print(f"  已注册技能：{len(self.skills)} 个")
        print("=" * 50)


# ====== 快速测试 ======
if __name__ == "__main__":
    agent = Agent(model="qwen2.5:7b")

    # 测试1：基础决策
    print("=" * 50)
    print("测试1：基础决策")
    print("=" * 50)
    skills = ["answer_question", "summarize_text", "translate"]
    
    tests = [
        "量子计算是什么？",
        "帮我总结一下这篇文章",
        "把这段话翻译成英文",
        "今天天气怎么样",  # 模糊输入
    ]
    
    for test_input in tests:
        result = agent.decide(test_input, skills)
        print(f"  输入：{test_input}")
        print(f"  决策：{result}")
        print()

    # 测试2：带路由的完整流程
    print("=" * 50)
    print("测试2：路由执行")
    print("=" * 50)
    
    # 定义几个简单的技能函数
    def skill_answer(text):
        return f"📝 [回答问题] 收到你的问题：{text}"

    def skill_summarize(text):
        return f"📋 [文本摘要] 正在为以下内容生成摘要：{text}"

    def skill_translate(text):
        return f"🌐 [翻译] 正在翻译：{text}"

    # 注册技能（带描述和示例，让 AI 更好识别）
    agent.register_skill(
        name="answer_question",
        func=skill_answer,
        description="回答用户的技术问题或知识查询",
        examples=["什么是装饰器", "Python 怎么读文件", "AI Agent 是什么"],
    )
    agent.register_skill(
        name="summarize_text",
        func=skill_summarize,
        description="总结或浓缩用户提供的文本内容",
        examples=["帮我总结一下", "缩写这段话", "提取这篇文章的要点"],
    )
    agent.register_skill(
        name="translate",
        func=skill_translate,
        description="翻译文本（默认中翻英）",
        examples=["翻译成英文", "这段话用英文怎么说", "translate this"],
    )
    
    agent.show_skills()
    
    # 测试路由
    print(agent.route("帮我翻译一下 Hello World"))
    print()
    print(agent.route("Python 的装饰器是什么？"))
