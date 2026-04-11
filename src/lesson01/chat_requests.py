import requests

def chat(prompt, model="qwen2.5:7b"):
    """单次对话"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

if __name__ == '__main__':
    result = chat("用一句话解释什么是Python")
    print(result)
