"""
Agent 类封装 - 第五课升级版：工具调用
在第四课（决策与路由）基础上，
新增 request_tool() 和 execute_tool_call() 方法。

依赖：pip install openai
"""

import json
import re
from openai import OpenAI
from typing import Optional, Generator, Any


# ==================== 工具函数（工具注册表之外） ====================

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    从文本中提取 JSON（第三课引入，第四课、第五课复用）

    因为模型经常会在 JSON 前后加一些废话，
    这个函数负责把真正的 JSON 找出来。

    尝试顺序：
    1. 直接把整个文本当 JSON 解析
    2. 用正则找第一个 { ... } 块
    """
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


def calculator(a, b, operation="add"):
    """
    计算器工具（第五课示例）

    Args:
        a: 第一个数字
        b: 第二个数字
        operation: 运算类型 - "add", "subtract", "multiply", "divide"
    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: division by zero",
    }
    fn = ops.get(operation)
    if fn is None:
        return f"Error: unknown operation '{operation}'"
    return fn(a, b)


def execute_tool(tool_name: str, arguments: dict) -> Any:
    """
    工具执行调度器

    根据工具名称调用对应的工具函数，传入参数。
    这是"执行层"——AI 永远无法绕过这个函数。

    Args:
        tool_name: 工具名称
        arguments: 工具参数字典

    Returns:
        工具执行结果
    """
    tools = {
        "calculator": calculator,
    }

    if tool_name not in tools:
        return f"Error: tool '{tool_name}' not found"

    return tools[tool_name](**arguments)


# ==================== Agent 类 ====================

class Agent:
    """
    AI Agent 类 - 第五课升级版

    新增功能（第五课）：
    - request_tool()：让 AI 请求工具调用（带参数提取）
    - execute_tool_call()：安全执行 AI 请求的工具调用
    - 工具注册表：动态添加/管理可用工具

    保留功能（第四课）：
    - decide()：从选项列表中做决策
    - decide_with_descriptions()：带描述的增强版决策
    - route()：意图识别 + 路由执行
    - register_skill()：注册技能

    保留功能（第二课、第三课）：
    - System Prompt 角色管理
    - 多轮对话 + 流式输出
    - generate_structured()（结构化输出）
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

        # 第四课：技能注册表（决策选项）
        self.skills: dict[str, dict] = {}

        # 第五课：工具注册表（可调用的工具）
        # 键 = 工具名称，值 = {func, description, parameters}
        self.tools: dict[str, dict] = {}

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

    # ==================== 第四课方法 ====================

    def decide(self, user_input: str, choices: list[str]) -> Optional[str]:
        """
        让 AI 从选项列表中选择一个动作（第四课核心方法）

        Args:
            user_input: 用户说的话
            choices: AI 可以选择的动作列表（名称）

        Returns:
            选中的动作名称，如果3次重试都失败则返回 None
        """
        options = "\n".join(f"- {choice}" for choice in choices)

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

    def decide_with_descriptions(self, user_input: str) -> Optional[str]:
        """
        带描述信息的决策（推荐方法）

        Returns:
            选中的技能名称，如果失败则返回 None
        """
        if not self.skills:
            return None

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
        """注册一个新技能（第四课）"""
        self.skills[name] = {
            "func": func,
            "description": description,
            "examples": examples or [],
        }
        print(f"✅ 已注册技能：{name}")
        if description:
            print(f"   描述：{description}")

    def route(self, user_input: str) -> str:
        """完整的意图识别 + 路由执行流程（第四课）"""
        if not self.skills:
            return "❌ 没有注册任何技能。"

        decision = self.decide_with_descriptions(user_input)

        if decision is None:
            return "❌ 无法理解你的意图，请换个说法试试。"

        if decision == "unknown":
            return "🤔 抱歉，我暂时不知道怎么帮你处理这个请求。"

        print(f"🎯 识别意图：{decision}")

        if decision in self.skills:
            try:
                return self.skills[decision]["func"](user_input)
            except Exception as e:
                return f"⚠️ 技能执行出错：{e}"

        return f"⚠️ 未知动作：{decision}"

    # ==================== 第五课新增方法 ====================

    def register_tool(
        self,
        name: str,
        func: callable,
        description: str,
        parameters: dict,
    ) -> None:
        """
        注册一个新工具（第五课）

        和 register_skill() 的区别：
        - skill 是"决策选项"——AI 选一个名字，你的代码执行
        - tool 是"函数调用"——AI 指定函数名 + 参数，你的代码执行

        Args:
            name: 工具名称（AI 通过这个名称调用）
            func: 工具函数，参数由 parameters 定义
            description: 工具描述（给 AI 看的，告诉它这个工具做什么）
            parameters: 参数定义，格式：
                {
                    "param_name": {"type": "number", "description": "参数说明", "required": True},
                    ...
                }
        """
        self.tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters,
        }
        print(f"🔧 已注册工具：{name}")
        print(f"   描述：{description}")

    def _build_tool_prompt(self) -> str:
        """
        根据已注册的工具，自动构建工具描述 prompt

        AI 需要知道有哪些工具、每个工具需要什么参数。
        这个方法把 self.tools 翻译成 AI 能理解的文字描述。
        """
        if not self.tools:
            return ""

        tool_lines = []
        for name, info in self.tools.items():
            line = f"- {name}：{info['description']}"
            params = info.get("parameters", {})
            if params:
                param_strs = []
                for pname, pinfo in params.items():
                    required = "必填" if pinfo.get("required") else "可选"
                    param_strs.append(f"{pname} ({pinfo.get('type', 'any')}, {required})")
                line += f"\n  参数：{', '.join(param_strs)}"
                for pname, pinfo in params.items():
                    line += f"\n    - {pname}：{pinfo.get('description', '')}"
            tool_lines.append(line)

        return "\n".join(tool_lines)

    def request_tool(self, user_input: str) -> Optional[dict]:
        """
        让 AI 请求工具调用（第五课核心方法）

        AI 分析用户输入，选择合适的工具并提取参数。
        输出格式：{"tool": "工具名", "arguments": {参数字典}}

        和第四课 decide() 的区别：
        - decide() 输出 {"decision": "名字"} —— 只有意图
        - request_tool() 输出 {"tool": "名字", "arguments": {...}} —— 意图 + 参数

        Args:
            user_input: 用户的请求

        Returns:
            工具调用规范，如果请求失败则返回 None
        """
        tools_desc = self._build_tool_prompt()

        if not tools_desc:
            print("⚠️ 没有注册任何工具，无法请求工具调用。")
            return None

        user_prompt = f"""你是一个工具调用助手。根据用户请求，选择合适的工具并提取参数。

可用工具：
{tools_desc}

规则：
1. 只能从上面的工具名称中选择，不能自己创造工具
2. 只返回有效的 JSON，不要有任何解释
3. JSON 格式：{{"tool": "工具名称", "arguments": {{参数字典}}}}

用户请求：{user_input}

请返回 JSON："""

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

            if parsed and "tool" in parsed and "arguments" in parsed:
                # 验证工具是否存在
                if parsed["tool"] in self.tools:
                    return parsed
                # 模糊匹配工具名
                for tool_name in self.tools:
                    if parsed["tool"].lower().replace("_", "").replace("-", "") == tool_name.lower().replace("_", "").replace("-", ""):
                        parsed["tool"] = tool_name
                        return parsed
                print(f"⚠️ AI 请求了不存在的工具：{parsed['tool']}")

        return None

    def execute_tool_call(self, tool_call: dict) -> Any:
        """
        执行 AI 请求的工具调用

        这是"执行层"——AI 的请求到这里就结束了，真正的执行由你的代码完成。

        Args:
            tool_call: 带 "tool" 和 "arguments" 的字典

        Returns:
            工具执行的结果
        """
        tool_name = tool_call.get("tool")
        arguments = tool_call.get("arguments", {})

        if tool_name not in self.tools:
            return f"Error: tool '{tool_name}' not found"

        try:
            result = self.tools[tool_name]["func"](**arguments)
            return result
        except TypeError as e:
            return f"Error: invalid arguments - {e}"
        except Exception as e:
            return f"Error: {e}"

    def show_tools(self) -> None:
        """打印所有已注册的工具"""
        if not self.tools:
            print("🔧 当前没有注册任何工具")
            return

        print("=" * 55)
        print("  🔧 已注册工具列表：")
        print("-" * 55)
        for i, (name, info) in enumerate(self.tools.items(), 1):
            print(f"  {i}. {name}")
            print(f"     描述：{info['description']}")
            params = info.get("parameters", {})
            if params:
                for pname, pinfo in params.items():
                    required = "必填" if pinfo.get("required") else "可选"
                    print(f"     - {pname} ({pinfo.get('type', 'any')}, {required})：{pinfo.get('description', '')}")
        print("=" * 55)

    # ==================== 通用方法 ====================

    def show_skills(self) -> None:
        """打印所有已注册的技能"""
        if not self.skills:
            print("📋 当前没有注册任何技能")
            return

        print("=" * 55)
        print("  📋 已注册技能列表：")
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
        print(f"  已注册工具：{len(self.tools)} 个")
        print("=" * 50)


# ====== 快速测试 ======
if __name__ == "__main__":
    agent = Agent(model="qwen2.5:7b")

    # ====== 测试第四课功能 ======
    print("=" * 50)
    print("测试1：第四课 - 基础决策 + 路由")
    print("=" * 50)

    def skill_answer(text):
        return f"📝 [回答问题] {text}"

    def skill_summarize(text):
        return f"📋 [摘要] {text[:30]}..."

    agent.register_skill("answer_question", skill_answer, "回答问题",
                          examples=["什么是装饰器", "AI Agent 是什么"])
    agent.register_skill("summarize_text", skill_summarize, "摘要总结",
                          examples=["帮我总结一下", "缩写这段话"])

    print(agent.route("量子计算是什么？"))
    print()

    # ====== 测试第五课功能 ======
    print("=" * 50)
    print("测试2：第五课 - 工具调用")
    print("=" * 50)

    # 注册计算器工具
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

    agent.show_tools()

    # 测试工具调用
    tests = [
        "What is 42 * 7?",
        "100 减去 37 等于多少",
        "25 除以 5",
    ]

    for test_input in tests:
        print(f"\n💬 用户：{test_input}")
        tool_call = agent.request_tool(test_input)
        if tool_call:
            print(f"🔧 AI 请求：{tool_call}")
            result = agent.execute_tool_call(tool_call)
            print(f"📊 执行结果：{result}")
        else:
            print("❌ 工具调用失败")
