"""
Agent 类封装 - 第二课：System Prompt 角色管理
支持动态切换角色、流式输出、多轮对话
依赖：pip install openai
"""

from openai import OpenAI
from typing import Optional, Generator


class Agent:
    """
    AI Agent 基础类
    
    功能：
    - 通过 System Prompt 控制模型角色和行为
    - 支持普通输出和流式输出（打字机效果）
    - 支持多轮对话（带历史记忆）
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        初始化 Agent
        
        Args:
            model: 模型名称，默认 qwen2.5:7b
            base_url: Ollama API 地址
            api_key: API Key（Ollama 填任意值即可）
            system_prompt: 系统提示词/角色设定
            temperature: 温度参数，越低回复越一致（0-2）
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or "你是一个有帮助的AI助手。"
        self.temperature = temperature
        self.history: list[dict] = []  # 多轮对话历史

    def set_role(self, system_prompt: str) -> None:
        """切换 Agent 角色，同时清空对话历史"""
        self.system_prompt = system_prompt
        self.history.clear()
        print(f"✅ 角色已切换：{system_prompt[:30]}...")

    def generate_with_role(self, user_input: str) -> str:
        """
        使用 System Prompt 生成回复（单轮，无历史）
        
        这就是第二课的核心方法 — 将角色设定作为 system 消息，
        用户问题作为 user 消息，一起发给模型。
        
        Args:
            user_input: 用户输入的问题
            
        Returns:
            模型的回复文本
        """
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
        """
        带历史记忆的多轮对话
        
        每次调用会把用户消息和模型回复都存入 history，
        下次对话时一起发给模型，实现上下文连贯。
        
        Args:
            user_input: 用户输入
            
        Returns:
            模型回复
        """
        # 构建消息列表：system + 历史消息 + 当前用户输入
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

        # 保存到历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        # 防止历史太长导致超出上下文限制（保留最近20轮）
        if len(self.history) > 40:
            self.history = self.history[-40:]

        return reply

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        流式输出 — 打字机效果
        
        逐字返回模型生成的 token，
        适合在命令行或聊天界面中使用。
        
        Yields:
            每次生成的一个 token 片段
        """
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

        # 流式结束后保存完整回复到历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_reply})

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
        print("=" * 50)


# ====== 快速测试 ======
if __name__ == "__main__":
    agent = Agent(model="qwen2.5:7b")

    question = "什么是 AI Agent？"

    # 角色1：编程老师
    agent.set_role("你是一位耐心的编程老师，用简单的话解释概念。")
    print("【编程老师】")
    print(agent.generate_with_role(question))
    print()

    # 角色2：大白话解释
    agent.set_role("你用最简单的大白话和日常比喻来解释事物")
    print("【大白话模式】")
    print(agent.generate_with_role(question))
    print()

    # 角色3：极简回答
    agent.set_role("你最多用2句话回答，绝不多说。")
    print("【极简模式】")
    print(agent.generate_with_role(question))
