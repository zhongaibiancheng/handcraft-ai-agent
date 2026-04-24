"""
评测框架 - 第十一课引入

智能体能力的回归测试系统。

职责：
  - EvalResult：单个评测用例的结果
  - EvalSuiteResult：评测套件的汇总结果
  - AgentEval：评测执行器（结构化输出、工具调用、记忆周期）
  - print_eval_report：打印评测报告

设计原则：
  - 纯 Python，零外部测试框架依赖
  - 结构化结果：每个结果完整记录输入、预期、实际、错误
  - 硬断言优先：先确保基础设施正确
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """单个评测用例的结果。"""
    passed: bool
    input: str
    expected: Any = None
    actual: Any = None
    error: str | None = None


@dataclass
class EvalSuiteResult:
    """运行评测套件的结果。"""
    name: str
    passed: int = 0
    failed: int = 0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """通过率。"""
        return self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0.0

    def add_result(self, result: EvalResult):
        """添加单个测试结果。"""
        self.results.append(result)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        """生成一行摘要。"""
        status = "✓ PASSED" if self.failed == 0 else "✗ FAILED"
        return f"{self.name}: {status} ({self.passed}/{self.passed + self.failed})"


class AgentEval:
    """智能体能力的回归测试。"""

    def __init__(self, agent):
        self.agent = agent

    def test_structured_output(self, cases: list[dict]) -> EvalSuiteResult:
        """
        测试结构化输出是否正确解析并匹配 schema。

        检查项：
        1. 输出是否是有效 JSON（硬断言）
        2. 必需字段是否都存在（硬断言）
        """
        suite = EvalSuiteResult(name="Structured Output")

        for case in cases:
            result = self.agent.generate_structured(case["input"], case["schema"])

            # 检查 1：得到有效 JSON 了吗？
            if result is None:
                suite.add_result(EvalResult(
                    passed=False,
                    input=case["input"],
                    error="Failed to parse JSON"
                ))
                continue

            # 检查 2：必需字段存在吗？
            missing = [f for f in case.get("must_have_fields", []) if f not in result]
            if missing:
                suite.add_result(EvalResult(
                    passed=False,
                    input=case["input"],
                    expected={"fields": case.get("must_have_fields", [])},
                    actual={"missing": missing},
                    error=f"Missing fields: {missing}"
                ))
                continue

            suite.add_result(EvalResult(
                passed=True,
                input=case["input"],
                actual=result
            ))

        return suite

    def test_tool_calls(self, cases: list[dict]) -> EvalSuiteResult:
        """
        测试工具调用是否选择了正确的工具。

        检查项：
        1. 工具调用是否返回非 None
        2. 工具名称是否匹配预期
        """
        suite = EvalSuiteResult(name="Tool Calls")

        for case in cases:
            tool_call = self.agent.request_tool(case["input"])

            # 检查 1：工具调用成功了吗？
            if tool_call is None:
                suite.add_result(EvalResult(
                    passed=False,
                    input=case["input"],
                    expected={"tool": case.get("expected_tool")},
                    error="No tool call returned"
                ))
                continue

            # 检查 2：工具名称匹配吗？
            actual_tool = tool_call.get("tool", "")
            expected_tool = case.get("expected_tool", "")
            if actual_tool.lower().replace("_", "").replace("-", "") != expected_tool.lower().replace("_", "").replace("-", ""):
                suite.add_result(EvalResult(
                    passed=False,
                    input=case["input"],
                    expected={"tool": expected_tool},
                    actual={"tool": actual_tool},
                    error=f"Wrong tool: expected '{expected_tool}', got '{actual_tool}'"
                ))
                continue

            suite.add_result(EvalResult(
                passed=True,
                input=case["input"],
                expected={"tool": expected_tool},
                actual={"tool": actual_tool, "arguments": tool_call.get("arguments", {})}
            ))

        return suite

    def test_memory_cycle(self, cases: list[dict]) -> EvalSuiteResult:
        """
        测试记忆存储和检索周期。

        检查项：
        1. 存储事实后，能否检索到
        2. 回答中是否包含关键信息
        """
        suite = EvalSuiteResult(name="Memory Cycle")

        for case in cases:
            # 清空记忆，确保干净状态
            self.agent.memory.clear()

            # 第一步：存储事实
            store_result = self.agent.run_with_memory(case["store_input"])

            if store_result is None:
                suite.add_result(EvalResult(
                    passed=False,
                    input=f"STORE: {case['store_input']}",
                    error="Memory store failed (run_with_memory returned None)"
                ))
                continue

            # 第二步：查询事实
            query_result = self.agent.run_with_memory(case["query_input"])

            if query_result is None:
                suite.add_result(EvalResult(
                    passed=False,
                    input=f"QUERY: {case['query_input']}",
                    expected={"contains": case.get("expected_in_response")},
                    error="Memory query failed (run_with_memory returned None)"
                ))
                continue

            # 检查回答中是否包含预期内容
            reply = query_result.get("reply", "")
            expected_text = case.get("expected_in_response", "")

            if expected_text.lower() in reply.lower():
                suite.add_result(EvalResult(
                    passed=True,
                    input=f"STORE: {case['store_input']} → QUERY: {case['query_input']}",
                    expected={"contains": expected_text},
                    actual={"reply": reply[:100]}
                ))
            else:
                suite.add_result(EvalResult(
                    passed=False,
                    input=f"STORE: {case['store_input']} → QUERY: {case['query_input']}",
                    expected={"contains": expected_text},
                    actual={"reply": reply[:100]},
                    error=f"Expected '{expected_text}' in reply, but not found"
                ))

        # 清空测试记忆
        self.agent.memory.clear()

        return suite

    def run_all(self, structured_cases=None, tool_cases=None, memory_cases=None) -> list:
        """运行所有评测套件。"""
        results = []

        if structured_cases:
            results.append(self.test_structured_output(structured_cases))

        if tool_cases:
            results.append(self.test_tool_calls(tool_cases))

        if memory_cases:
            results.append(self.test_memory_cycle(memory_cases))

        return results


def print_eval_report(results: list) -> None:
    """打印评测报告。"""
    print("=" * 50)
    print("EVAL REPORT")
    print("=" * 50)
    print()

    total_passed = 0
    total_failed = 0

    for suite in results:
        print(suite.summary())
        total_passed += suite.passed
        total_failed += suite.failed

        # 只打印失败的用例详情
        for r in suite.results:
            if not r.passed:
                input_display = r.input[:50] + "..." if len(r.input) > 50 else r.input
                print(f"  ✗ Input: {input_display}")
                if r.expected:
                    print(f"    Expected: {r.expected}")
                if r.actual:
                    print(f"    Actual: {r.actual}")
                if r.error:
                    print(f"    Error: {r.error}")
        print()

    print("-" * 50)
    total = total_passed + total_failed
    if total > 0:
        status = "✓ ALL PASSED" if total_failed == 0 else f"✗ {total_failed} FAILED"
        print(f"Overall: {status} ({total_passed}/{total})")
    else:
        print("Overall: No test cases executed")
    print("=" * 50)
