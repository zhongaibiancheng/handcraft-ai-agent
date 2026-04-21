"""
共享工具函数

第三课引入的 extract_json_from_text，供 planner.py 等模块复用。
"""

import json
import re
from typing import Optional


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    从文本中提取 JSON（第三课引入）。

    尝试顺序：
    1. 直接把整个文本当 JSON 解析
    2. 用正则找第一个 { ... } 块

    Args:
        text: LLM 返回的文本

    Returns:
        解析后的字典，失败返回 None
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
