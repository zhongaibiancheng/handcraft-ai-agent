import ollama
from extract_json import extract_json_from_text


def generate_structured(
    user_input: str,
    schema: str,
    system_prompt: str = "",
    model: str = "qwen2.5:7b",
    max_retries: int = 3,
) -> dict | None:
    """
    调用本地大模型，生成结构化 JSON 输出，带重试机制。

    Args:
        user_input:   用户输入
        schema:       JSON schema 描述（字符串形式）
        system_prompt: 基础系统提示词（可选）
        model:        Ollama 模型名称
        max_retries:  最大重试次数

    Returns:
        解析后的字典，全部失败返回 None
    """
    # 把 JSON 要求注入 System Prompt
    full_system_prompt = f"""{system_prompt}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON. No explanations, no markdown, no extra text.
2. Start your response with {{ and end with }}.
3. Follow this schema exactly:
{schema}
""".strip()

    messages = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_input},
    ]

    for attempt in range(1, max_retries + 1):
        print(f"[第 {attempt} 次尝试]")

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0},  # 降低随机性，输出更稳定
        )

        raw_text = response["message"]["content"]
        parsed = extract_json_from_text(raw_text)

        if parsed is not None:
            print(f"✓ 解析成功")
            return parsed
        else:
            print(f"✗ 解析失败，原始输出：{raw_text[:100]}")

    print("❌ 全部重试失败，返回 None")
    return None


# ---- 测试 ----
if __name__ == "__main__":
    schema = '''
{
    "sentiment": "正面" | "负面" | "中性",
    "confidence": "high" | "medium" | "low"
}
'''

    result = generate_structured(
        user_input="今天天气很好，我要出去爬山！",
        schema=schema,
        system_prompt="你是一个情感分析助手。",
    )

    print(f"\n最终结果: {result}")
