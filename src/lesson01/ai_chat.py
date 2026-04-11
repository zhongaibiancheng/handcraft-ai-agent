import requests

def chat(prompt, model="qwen2.5:7b"):
    """发送消息给本地模型"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

if __name__ == "__main__":
    print("=" * 50)
    print("本地 AI 助手已启动")
    print("模型：Qwen2.5:7B（运行在你自己电脑上）")
    print("输入 q 退出")
    print("=" * 50)
    print()
    
    while True:
        user_input = input("你: ")
        
        if user_input.lower() == "q":
            print("再见！")
            break
        
        reply = chat(user_input)
        print(f"AI: {reply}\n")