"""
Agent 类封装 - 第八课升级版：规划系统
在第七课（记忆系统）基础上，
新增 agent/planner.py 模块和 create_plan() / execute_plan() 方法。

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
    """计算器工具（第五课示例）"""
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
    """工具执行调度器（第五课）"""
    tools = {"calculator": calculator}
    if tool_name not in tools:
        return f"Error: tool '{tool_name}' not found"
    return tools[tool_name](**arguments)


# ==================== 状态类（第六课引入） ====================

class AgentState:
    """智能体状态追踪（第六课引入）"""

    def __init__(self):
        self.steps: int = 0
        self.done: bool = False
        self.results: list[dict] = []
        self.current_plan: Optional[dict] = None  # 第八课新增：存储当前计划

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "done": self.done,
            "results": self.results[-3:],
        }

    def add_result(self, result: dict) -> None:
        self.results.append(result)

    def increment_step(self) -> None:
        self.steps += 1

    def mark_done(self) -> None:
        self.done = True

    def reset(self) -> None:
        self.steps = 0
        self.done = False
        self.results = []
        self.current_plan = None


# ==================== 记忆类（第七课引入） ====================

class AgentMemory:
    """
    智能体记忆系统（第七课引入）

    最简单的记忆实现：一个字符串列表。
    存储 AI 认为值得长期记住的事实。

    设计原则：
    - AI 控制存什么（通过 save_to_memory 字段）
    - 你控制怎么存（通过 add/remove/clear 方法）
    - 透明可控，没有黑盒
    """

    def __init__(self):
        self.memories: list[str] = []

    def add(self, fact: str) -> None:
        """存储一条记忆（自动去重）"""
        if fact and fact.strip() and fact.strip() not in self.memories:
            self.memories.append(fact.strip())
            print(f"🧠 存入记忆：{fact.strip()}")

    def get_all(self) -> list[str]:
        """获取所有记忆（返回副本）"""
        return self.memories.copy()

    def search(self, keyword: str) -> list[str]:
        """按关键词搜索记忆（简单版）"""
        keyword = keyword.lower()
        return [m for m in self.memories if keyword in m.lower()]

    def remove(self, fact: str) -> bool:
        """删除一条记忆"""
        fact = fact.strip()
        if fact in self.memories:
            self.memories.remove(fact)
            print(f"🗑️ 删除记忆：{fact}")
            return True
        return False

    def clear(self) -> None:
        """清空所有记忆"""
        self.memories.clear()
        print("🧹 所有记忆已清空")

    def count(self) -> int:
        """记忆条数"""
        return len(self.memories)

    def show(self) -> None:
        """打印所有记忆"""
        if not self.memories:
            print("🧠 当前没有记忆")
            return

        print("=" * 55)
        print(f"  🧠 记忆列表（共 {len(self.memories)} 条）：")
        print("-" * 55)
        for i, m in enumerate(self.memories, 1):
            print(f"  {i}. {m}")
        print("=" * 55)


# ==================== Agent 类 ====================

class Agent:
    """
    AI Agent 类 - 第八课升级版

    新增功能（第八课）：
    - create_plan()：根据目标生成执行计划
    - execute_plan()：按计划逐步执行
    - agent/planner.py：独立规划器模块

    保留功能（第七课）：
    - AgentMemory：跨对话记忆系统
    - run_with_memory()：带记忆的对话执行

    保留功能（第六课）：
    - AgentState：状态追踪
    - agent_step()：单步执行
    - run_loop()：多步循环引擎

    保留功能（第五课）：
    - request_tool() / execute_tool_call()
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

        # 第七课：记忆系统
        self.memory = AgentMemory()

    # ==================== 第二课方法 ====================

    def set_role(self, system_prompt: str) -> None:
        """切换 Agent 角色，同时清空对话历史"""
        self.system_prompt = system_prompt
        self.history.clear()
        print(f"✅ 角色已切换：{system_prompt[:30]}...")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """底层生成方法（供 planner.py 等外部模块调用）"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

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

    # ==================== 第六课方法 ====================

    def agent_step(self, user_input: str) -> Optional[dict]:
        """执行智能体循环的一步：观察→决策→执行（第六课核心方法）"""
        state_dict = self.state.to_dict()

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
                valid_actions = ["analyze", "research", "summarize", "answer", "done"]
                if parsed["action"] not in valid_actions:
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
        """运行多步智能体循环（第六课核心方法）"""
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
                print(f"   ❌ 步骤失败，退出循环")
                break

            print()

        if not self.state.done and self.state.steps >= max_steps:
            print(f"⚠️ 达到最大步数 {max_steps}，强制结束")

        print(f"\n📊 循环结束，共执行 {len(results)} 步")
        return results

    # ==================== 第七课方法 ====================

    def run_with_memory(self, user_input: str) -> Optional[dict]:
        """使用记忆上下文运行智能体（第七课核心方法）"""
        memory_context = self.memory.get_all()

        if memory_context:
            memory_str = "你记住了以下关于用户的信息：\n" + "\n".join(
                f"- {item}" for item in memory_context
            )
        else:
            memory_str = "你目前没有关于用户的记忆。"

        user_prompt = f"""你是一个有记忆能力的智能体助手。根据用户输入和你记住的信息来回复。

{memory_str}

规则：
1. 只返回有效的 JSON
2. 不要任何解释，不要 Markdown
3. 直接以 {{ 开头，以 }} 结尾
4. 如果用户告诉你新信息（比如名字、偏好、项目信息），请保存到记忆中
5. 如果用户问到你记得的信息，请使用记忆来回答
6. JSON 格式：{{"reply": "你的回复内容", "save_to_memory": "要记住的事实" 或 null}}

示例：
- 用户说"我叫小明" → {{"reply": "你好，小明！", "save_to_memory": "用户的名字是小明"}}
- 用户问"我叫什么"且你记得"用户的名字是小明" → {{"reply": "你叫小明", "save_to_memory": null}}
- 用户说"帮我写个函数" → {{"reply": "好的...", "save_to_memory": null}}

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

            if parsed and "reply" in parsed:
                if parsed.get("save_to_memory"):
                    self.memory.add(parsed["save_to_memory"])

                return parsed

        return None

    # ==================== 第八课新增方法 ====================

    def create_plan(self, goal: str) -> Optional[dict]:
        """
        生成实现目标的计划（第八课核心方法）。

        委托给 agent/planner.py 模块，职责分离。

        Args:
            goal: 要实现的目标

        Returns:
            包含 "steps" 列表的计划字典，失败则返回 None
        """
        from agent.planner import create_plan as plan_fn

        plan = plan_fn(self, goal)

        if plan:
            self.state.current_plan = plan
            print(f"📋 计划生成完成，共 {len(plan['steps'])} 个步骤")

        return plan

    def execute_plan(self, plan: dict) -> list[dict]:
        """
        逐步执行计划（第八课核心方法）。

        Args:
            plan: 包含 "steps" 列表的计划字典

        Returns:
            每步的执行结果列表
        """
        if not plan or "steps" not in plan:
            print("❌ 无效的计划")
            return []

        results = []
        total = len(plan["steps"])

        print(f"\n🚀 开始执行计划（共 {total} 个步骤）\n")

        for i, step in enumerate(plan["steps"], 1):
            print(f"--- 步骤 {i}/{total} ---")
            print(f"   内容：{step}")

            result = {
                "step_num": i,
                "step": step,
                "executed": True,
                "status": "completed"
            }
            results.append(result)
            self.state.increment_step()

            print(f"   状态：✅ 已执行\n")

        return results

    # ==================== 通用方法 ====================

    def show_state(self) -> None:
        """打印当前智能体状态"""
        state = self.state
        print("=" * 55)
        print("  📊 智能体状态：")
        print("-" * 55)
        print(f"  步骤数：{state.steps}")
        print(f"  是否完成：{'✅ 是' if state.done else '❌ 否'}")
        print(f"  历史结果：{len(state.results)} 条记录")
        if state.current_plan:
            print(f"  当前计划：{len(state.current_plan.get('steps', []))} 个步骤")
        for i, r in enumerate(state.results, 1):
            print(f"    {i}. {r}")
        print("=" * 55)

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
        print(f"  循环步数：{self.state.steps}")
        print(f"  记忆条数：{self.memory.count()} 条")
        if self.state.current_plan:
            print(f"  当前计划：{len(self.state.current_plan.get('steps', []))} 个步骤")
        print("=" * 55)


# ====== 快速测试 ======
if __name__ == "__main__":
    agent = Agent(model="qwen2.5:7b")

    print("=" * 55)
    print("  第八课测试：规划系统")
    print("=" * 55)
    print()

    # 生成计划
    print("--- 生成计划 ---")
    plan = agent.create_plan("写一篇关于 AI Agent 的技术博客")

    if plan:
        print("\n📋 计划内容：")
        for i, step in enumerate(plan["steps"], 1):
            print(f"  {i}. {step}")

        # 执行计划
        results = agent.execute_plan(plan)
        print(f"\n📊 执行完成，共 {len(results)} 个步骤")
    else:
        print("❌ 计划生成失败")
