import json
import re


def extract_json_from_text(text: str) -> dict | None:
    """
    从模型的原始输出中提取 JSON。

    处理以下几种情况：
    - 标准 JSON：{"key": "value"}
    - 前后有废话：好的！以下是结果：{"key": "value"} 希望有帮助。
    - Markdown 代码块包裹：```json\n{"key": "value"}\n```

    Returns:
        解析后的字典，解析失败返回 None
    """
    if not text:
        return None

    # 先尝试直接解析（最理想的情况）
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 处理 Markdown 代码块：```json ... ``` 或 ``` ... ```
    md_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    md_match = re.search(md_pattern, text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 从文本中找第一个 { ... } 块
    brace_pattern = r"\{[\s\S]*?\}"
    brace_match = re.search(brace_pattern, text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---- 测试 ----
if __name__ == "__main__":
    cases = [
        # 标准 JSON
        '{"sentiment": "正面", "confidence": "high"}',
        # 前后有废话
        '好的！以下是结果：{"sentiment": "正面", "confidence": "high"} 希望对你有帮助。',
        # Markdown 代码块
        '```json\n{"sentiment": "负面", "confidence": "low"}\n```',
        # 解析失败
        "这是一个无法解析的回答。",
    ]

    for case in cases:
        result = extract_json_from_text(case)
        print(f"输入: {case[:50]}...")
        print(f"结果: {result}")
        print()
