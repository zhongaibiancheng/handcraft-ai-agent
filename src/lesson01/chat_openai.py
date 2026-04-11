from openai import OpenAI

# Ollama 的 OpenAI 兼容端点
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama 不需要真正的 key，随便填
)

def chat(prompt, model="qwen2.5:7b"):
    """使用 OpenAI SDK 调用本地模型"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    # 测试
    result = chat("什么是大语言模型？")
    print(result)