"""
第三课完整项目：命令行情感分析工具

功能：
- 用户输入任意一句话
- 返回情感分类（正面/负面/中性）+ 置信度
- 自动重试，保证输出可靠

运行方式：
    python sentiment_analysis.py
"""

import ollama
import json
import re


# ============================================================
# 工具函数：从文本中提取 JSON
# ============================================================

def extract_json_from_text(text: str) -> dict | None:
    """从模型原始输出中提取 JSON"""
    if not text:
        return None
    # 直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Markdown 代码块
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 找第一个 {...}
    brace_match = re.search(r"\{[\s\S]*?\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    return None


# ============================================================
# 核心函数：结构化生成（带重试）
# ============================================================

SYSTEM_PROMPT = """你是一个情感分析助手。

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON. No explanations, no markdown, no extra text.
2. Start your response with { and end with }.
3. Follow this schema exactly:
{
    "sentiment": "正面" | "负面" | "中性",
    "confidence": "high" | "medium" | "low",
    "reason": string  // 一句话说明原因
}
"""

SCHEMA = """
{
    "sentiment": "正面" | "负面" | "中性",
    "confidence": "high" | "medium" | "low",
    "reason": string
}
"""


def validate(data: dict) -> bool:
    """验证字段是否合法"""
    if "sentiment" not in data or "confidence" not in data:
        return False
    if data["sentiment"] not in ["正面", "负面", "中性"]:
        return False
    if data["confidence"] not in ["high", "medium", "low"]:
        return False
    return True


def analyze_sentiment(text: str, model: str = "qwen2.5:7b", max_retries: int = 3) -> dict | None:
    """分析情感，带重试"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    for attempt in range(1, max_retries + 1):
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0},
        )
        raw = response["message"]["content"]
        parsed = extract_json_from_text(raw)

        if parsed and validate(parsed):
            return parsed

    return None


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 50)
    print("情感分析工具（基于 Qwen2.5:7B 本地模型）")
    print("输入 q 退出")
    print("=" * 50)
    print()

    confidence_map = {
        "high": "高",
        "medium": "中",
        "low": "低",
    }

    while True:
        user_input = input("请输入一句话：").strip()

        if user_input.lower() == "q":
            print("再见！")
            break

        if not user_input:
            continue

        print("分析中...")
        result = analyze_sentiment(user_input)

        if result:
            sentiment = result.get("sentiment", "未知")
            confidence = confidence_map.get(result.get("confidence", ""), "未知")
            reason = result.get("reason", "")
            print(f"\n情感：{sentiment}（置信度：{confidence}）")
            print(f"原因：{reason}\n")
        else:
            print("❌ 分析失败，请重试\n")


if __name__ == "__main__":
    main()
