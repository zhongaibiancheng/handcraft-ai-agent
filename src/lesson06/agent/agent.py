"""
Agent 类封装 - 第六课升级版：智能体循环
在第五课（工具调用）基础上，
新增 AgentState、agent_step() 和 run_loop() 方法。

依赖：pip install openai
"""

import json
import re
from openai import OpenAI
from typing import Optional, Generator, Any


# ==================== 工具函数 ====================

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    从文本中提取 JSON（第三课引入，后续课程复用）

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
    工具执行调度器（第五课）
    """
    tools = {
        "calculator": calculator,
    }

    if tool_name not in tools:
        return f"Error: tool '{tool_name}' not found"

    return tools[tool_name](**arguments)


# ==================== 状态类（第六课新增） ====================

class AgentState:
    """
    智能体状态追踪（第六课引入）

    跟踪智能体在循环中的进度：
    - 执行了多少步
    - 是否已完成
    - 每步的结果是什么
    """

    def __init__(self):
        self.steps: int = 0
        self.done: bool = False
        self.results: list[dict] = []

    def to_dict(self) -> dict:
        """将状态转换为字典（用于拼入 Prompt）"""
        return {
            "steps": self.steps,
            "done": self.done,
            "results": self.results[-3:],  # 只传最近3步，防止 prompt 过长
        }

    def add_result(self, result: dict) -> None:
        """记录一步的结果"""
        self.results.append(result)

    def increment_step(self) -> None:
        """步数 +1"""
        self.steps += 1

    def mark_done(self) -> None:
        """标记任务完成"""
        self.done = True

    def reset(self) -> None:
        """重置状态（开始新一轮循环前调用）"""
        self.steps = 0
        self.done = False
        self.results = []


# ==================== Agent 类 ====================

class Agent:
    """
    AI Agent 类 - 第六课升级版

    新增功能（第六课）：
    - AgentState：状态追踪（步骤计数、完成标志、结果累积）
    - agent_step()：单步执行（观察→决策→执行）
    - run_loop()：多步循环引擎（带终止条件）

    保留功能（第五课）：
    - request_tool()：让 AI 请求工具调用
    - execute_tool_call()：安全执行工具调用
    - 工具注册表

    保留功能（第四课）：
    - decide() / decide_with_descriptions() / route()
    - 技能注册表

    保留功能（第二课、第三课）：
    - System Prompt 角色管理
    - 多轮对话 + 流式输出
    - generate_structured()
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

        # 第四课：技能注册表
        self.skills: dict[str, dict] = {}

        # 第五课：工具注册表
        self.tools: dict[str, dict] = {}

        # 第六课：智能体状态
        self.state = AgentState()

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
        """强制输出结构化 JSON（第三课核心方法）"""
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
        """让 AI 从选项列表中选择一个动作（第四课核心方法）"""
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
        """带描述信息的决策（推荐方法）"""
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

    # ==================== 第五课方法 ====================

    def register_tool(
        self,
        name: str,
        func: callable,
        description: str,
        parameters: dict,
    ) -> None:
        """注册一个新工具（第五课）"""
        self.tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters,
        }
        print(f"🔧 已注册工具：{name}")
        print(f"   描述：{description}")

    def _build_tool_prompt(self) -> str:
        """根据已注册的工具，自动构建工具描述 prompt"""
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
        """让 AI 请求工具调用（第五课核心方法）"""
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
                if parsed["tool"] in self.tools:
                    return parsed
                for tool_name in self.tools:
                    if parsed["tool"].lower().replace("_", "").replace("-", "") == tool_name.lower().replace("_", "").replace("-", ""):
                        parsed["tool"] = tool_name
                        return parsed
                print(f"⚠️ AI 请求了不存在的工具：{parsed['tool']}")

        return None

    def execute_tool_call(self, tool_call: dict) -> Any:
        """执行 AI 请求的工具调用"""
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

    # ==================== 第六课新增方法 ====================

    def agent_step(self, user_input: str) -> Optional[dict]:
        """
        执行智能体循环的一步：观察→决策→执行。

        第六课核心方法。

        每次 agent_step 做三件事：
        1. 观察：读取当前状态（步骤数、历史动作）
        2. 决策：让 LLM 选择下一步动作
        3. 执行：更新状态（步数+1，记录结果）

        Args:
            user_input: 用户输入或系统观察

        Returns:
            动作决策，如果步骤失败则返回 None
        """
        state_dict = self.state.to_dict()

        # 构建历史动作摘要
        history_summary = ""
        if state_dict.get("results"):
            history_lines = []
            for i, r in enumerate(state_dict["results"], 1):
                action = r.get("action", "?")
                reason = r.get("reason", "")
                history_lines.append(f"  步骤{i}: 动作={action}, 原因={reason}")
            history_summary = "\n之前的动作：\n" + "\n".join(history_lines)

        user_prompt = f"""你是一个智能体助手。你需要根据当前状态，决定下一步该做什么。

当前状态：步骤={state_dict.get('steps', 0)}, 已完成={state_dict.get('done', False)}
{history_summary}

可用动作：
- analyze：分析用户输入或已有结果
- research：深入研究某个方面
- summarize：总结已获得的信息
- answer：给出最终答案
- done：任务已完成，可以结束

规则：
1. 只返回有效的 JSON
2. 不要任何解释，不要 Markdown
3. 直接以 {{ 开头，以 }} 结尾
4. JSON 格式：{{"action": "动作名称", "reason": "选择该动作的原因"}}

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

            if parsed and "action" in parsed:
                # 验证动作是否在可用列表中
                valid_actions = ["analyze", "research", "summarize", "answer", "done"]
                if parsed["action"] not in valid_actions:
                    # 模糊匹配
                    for va in valid_actions:
                        if parsed["action"].lower().replace("_", "").replace("-", "") == va:
                            parsed["action"] = va
                            break

                if "reason" not in parsed:
                    parsed["reason"] = f"执行动作: {parsed['action']}"
                self.state.increment_step()
                self.state.add_result(parsed)
                return parsed

        return None

    def run_loop(self, user_input: str, max_steps: int = 5) -> list[dict]:
        """
        运行多步智能体循环（第六课核心方法）。

        循环结构：
        while 未完成 and 步数 < 最大步数:
            执行一步
            如果 AI 说 done → 标记完成，退出
            如果解析失败 → 强制退出

        三个终止条件（双重保险）：
        1. AI 主动说 done → 正常退出
        2. 步数达到 max_steps → 安全限制
        3. 解析失败 → 兜底退出

        Args:
            user_input: 初始用户输入
            max_steps: 最大执行步数（安全限制）

        Returns:
            每步的动作结果列表
        """
        self.state.reset()
        results = []

        print(f"🔄 启动智能体循环（最多 {max_steps} 步）")
        print(f"   用户输入：{user_input}\n")

        while not self.state.done and self.state.steps < max_steps:
            step_num = self.state.steps + 1
            print(f"--- 步骤 {step_num} ---")

            action = self.agent_step(user_input)

            if action:
                action_name = action.get("action", "?")
                reason = action.get("reason", "")
                print(f"   动作：{action_name}")
                print(f"   原因：{reason}")
                results.append(action)

                if action_name == "done":
                    self.state.mark_done()
                    print(f"   ✅ 智能体主动结束")
                    break
            else:
                print(f"   ❌ 步骤失败（LLM 返回无法解析），退出循环")
                break

            print()

        if not self.state.done and self.state.steps >= max_steps:
            print(f"⚠️ 达到最大步数 {max_steps}，强制结束")

        print(f"\n📊 循环结束，共执行 {len(results)} 步")
        return results

    def show_state(self) -> None:
        """打印当前智能体状态"""
        state = self.state
        print("=" * 55)
        print("  📊 智能体状态：")
        print("-" * 55)
        print(f"  步骤数：{state.steps}")
        print(f"  是否完成：{'✅ 是' if state.done else '❌ 否'}")
        print(f"  历史结果：{len(state.results)} 条记录")
        for i, r in enumerate(state.results, 1):
            print(f"    {i}. {r}")
        print("=" * 55)

    # ==================== 通用方法 ====================

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
        print("=" * 55)
        print(f"  模型：{self.model}")
        print(f"  角色：{self.system_prompt}")
        print(f"  温度：{self.temperature}")
        print(f"  历史轮数：{len(self.history) // 2}")
        print(f"  已注册技能：{len(self.skills)} 个")
        print(f"  已注册工具：{len(self.tools)} 个")
        print(f"  循环步数：{self.state.steps} / 最大不限")
        print(f"  循环状态：{'已完成' if self.state.done else '进行中'}")
        print("=" * 55)


# ====== 快速测试 ======
if __name__ == "__main__":
    agent = Agent(model="qwen2.5:7b")

    print("=" * 55)
    print("  第六课测试：智能体循环")
    print("=" * 55)
    print()

    # 测试智能体循环
    results = agent.run_loop("帮我分析一下 Python 的优点和缺点", max_steps=3)

    print("\n--- 结果汇总 ---")
    for i, result in enumerate(results, 1):
        action = result.get("action", "unknown")
        reason = result.get("reason", "无原因")
        print(f"  步骤{i}: [{action}] {reason}")

    print()
    agent.show_state()
