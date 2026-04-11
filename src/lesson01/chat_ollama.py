import ollama

def chat(prompt, model="qwen2.5:7b"):
    """使用官方 SDK 调用"""
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

if __name__ == '__main__':
    # 测试
    result = chat("你好，请介绍一下你自己")
    print(result)