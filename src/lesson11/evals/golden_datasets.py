"""
黄金数据集 - 第十一课引入

评测的"真值来源"——已知良好的测试用例，必须始终通过。
如果黄金用例失败，是 Agent 坏了，不是测试坏了。

设计原则：
  - 带 Example 的多行 schema（比单行更稳定）
  - 覆盖多种输入（英文、中文、特殊字符）
  - 具体断言（不只是"它工作"，而是"这个字段存在"）
  - 与 Prompt 一起版本控制
"""

# ==================== 结构化输出黄金数据集 ====================

STRUCTURED_OUTPUT_GOLDEN = [
    {
        "input": "Explain quantum computing in one sentence",
        "schema": """{
  "topic": "the topic name as a string",
  "difficulty": "beginner" or "intermediate" or "advanced"
}

Example: {"topic": "machine learning", "difficulty": "intermediate"}""",
        "must_have_fields": ["topic", "difficulty"]
    },
    {
        "input": "What does 'hello world' mean in programming?",
        "schema": """{
  "explanation": "brief explanation as a string",
  "language": "the programming language it's most associated with"
}""",
        "must_have_fields": ["explanation", "language"]
    },
    {
        "input": "Describe the uses of Python",
        "schema": """{
  "language": "language name",
  "use_cases": "list of main use cases",
  "popularity": "beginner" or "intermediate" or "advanced"
}""",
        "must_have_fields": ["language", "use_cases", "popularity"]
    },
    {
        "input": "What is artificial intelligence?",
        "schema": """{
  "topic": "the topic",
  "summary": "one sentence summary",
  "difficulty": "beginner" or "intermediate" or "advanced"
}""",
        "must_have_fields": ["topic", "summary", "difficulty"]
    },
]

# ==================== 工具调用黄金数据集 ====================

TOOL_CALL_GOLDEN = [
    {
        "input": "What is 42 * 7?",
        "expected_tool": "calculator",
        "expected_args": {"operation": "multiply"}
    },
    {
        "input": "Calculate 100 divided by 4",
        "expected_tool": "calculator",
        "expected_args": {"operation": "divide"}
    },
    {
        "input": "What's the square root of 144?",
        "expected_tool": "calculator",
        "expected_args": {"operation": "sqrt"}
    },
    {
        "input": "Add 25 and 75",
        "expected_tool": "calculator",
        "expected_args": {"operation": "add"}
    },
    {
        "input": "Subtract 30 from 100",
        "expected_tool": "calculator",
        "expected_args": {"operation": "subtract"}
    },
]

# ==================== 记忆周期黄金数据集 ====================

MEMORY_GOLDEN = [
    {
        "store_input": "My name is Alice",
        "query_input": "What's my name?",
        "expected_in_response": "Alice"
    },
    {
        "store_input": "I live in Shenzhen, China",
        "query_input": "Where do I live?",
        "expected_in_response": "Shenzhen"
    },
    {
        "store_input": "My favorite programming language is Python",
        "query_input": "What's my favorite language?",
        "expected_in_response": "Python"
    },
]
