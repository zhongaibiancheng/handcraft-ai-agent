import requests
import json

def chat_stream(prompt, model="qwen2.5:7b"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        },
        stream=True
    )

    response.raise_for_status()

    print("AI: ", end="", flush=True)

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))
        content = data.get("message", {}).get("content", "")
        print(content, end="", flush=True)

        if data.get("done", False):
            break

    print()


if __name__ == '__main__':
    chat_stream("用 5 句话介绍一下 Python")