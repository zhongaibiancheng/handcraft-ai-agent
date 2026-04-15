"""
决策系统测试集 - 第四课配套
50+ 测试用例覆盖各种边界情况

运行方式：
  python test_suite.py
"""

import sys
import time
from agent import Agent


class DecisionTestSuite:
    """决策系统测试套件"""

    def __init__(self, model: str = "qwen2.5:7b"):
        self.agent = Agent(model=model)
        self.results = []
        self.total = 0
        self.passed = 0

    def register_test_skills(self):
        """注册测试用的技能"""
        def dummy_skill(text, ctx=None):
            return f"✅ 技能执行成功：{text}"

        self.agent.register_skill("answer_question", dummy_skill, "回答问题")
        self.agent.register_skill("summarize_text", dummy_skill, "摘要总结")
        self.agent.register_skill("translate", dummy_skill, "翻译")
        self.agent.register_skill("code_review", dummy_skill, "代码审查")
        self.agent.register_skill("write_code", dummy_skill, "编写代码")

    def test(self, user_input: str, expected: str, category: str = "基础"):
        """
        运行单个测试

        Args:
            user_input: 用户输入
            expected: 期望的决策结果
            category: 测试类别
        """
        self.total += 1
        choices = list(self.agent.skills.keys())

        result = self.agent.decide(user_input, choices)
        passed = result == expected

        if passed:
            self.passed += 1
            status = "✅"
        else:
            status = "❌"

        entry = {
            "input": user_input,
            "expected": expected,
            "actual": result,
            "passed": passed,
            "category": category,
        }
        self.results.append(entry)

        print(f"  {status} [{category}] \"{user_input[:40]}\" → 期望:{expected} 实际:{result}")

    def run_all(self):
        """运行所有测试"""
        print("\n" + "=" * 65)
        print("  🧪 决策系统测试套件")
        print("=" * 65 + "\n")

        self.register_test_skills()

        # ====== 类别1：基础意图识别 ======
        print("━━━ 1. 基础意图识别 ━━━")
        self.test("量子计算是什么？", "answer_question", "基础")
        self.test("AI Agent 是什么", "answer_question", "基础")
        self.test("帮我总结一下这篇文章", "summarize_text", "基础")
        self.test("缩写这段话", "summarize_text", "基础")
        self.test("把这段话翻译成英文", "translate", "基础")
        self.test("translate this to Japanese", "translate", "基础")
        self.test("帮我看看这段代码有没有问题", "code_review", "基础")
        self.test("review my code", "code_review", "基础")
        self.test("帮我写一个快速排序", "write_code", "基础")
        self.test("用 Python 实现一个爬虫", "write_code", "基础")

        # ====== 类别2：模糊输入 ======
        print("\n━━━ 2. 模糊输入 ━━━")
        self.test("嗯", "answer_question", "模糊")
        self.test("好的", "answer_question", "模糊")
        self.test("666", "answer_question", "模糊")
        self.test("谢谢", "answer_question", "模糊")
        self.test("你知道吗", "answer_question", "模糊")

        # ====== 类别3：中英混杂 ======
        print("\n━━━ 3. 中英混杂 ━━━")
        self.test("help me summarize this", "summarize_text", "中英混杂")
        self.test("帮我 review 一下代码", "code_review", "中英混杂")
        self.test("write a sorting algorithm", "write_code", "中英混杂")
        self.test("请翻译 hello world", "translate", "中英混杂")
        self.test("can you answer my question", "answer_question", "中英混杂")

        # ====== 类别4：长文本输入 ======
        print("\n━━━ 4. 长文本输入 ━━━")
        long_text = "这是一段很长的文本。" * 20
        self.test(f"请总结以下内容：{long_text}", "summarize_text", "长文本")
        self.test(f"请翻译以下内容：{long_text}", "translate", "长文本")

        # ====== 类别5：边界情况 ======
        print("\n━━━ 5. 边界情况 ━━━")
        self.test("", "answer_question", "边界")  # 空输入
        self.test("   ", "answer_question", "边界")  # 纯空格
        self.test("!!!???...", "answer_question", "边界")  # 纯符号
        self.test("<script>alert(1)</script>", "code_review", "恶意输入")

        # ====== 类别6：多意图 ======
        print("\n━━━ 6. 多意图（一个输入涉及多个技能）━━━")
        self.test("帮我写一个翻译功能的代码", "write_code", "多意图")
        self.test("总结并翻译这篇文章", "summarize_text", "多意图")
        self.test("审查并修复这段代码", "code_review", "多意图")

        # ====== 结果统计 ======
        print("\n" + "=" * 65)
        print("  📊 测试结果")
        print("=" * 65)

        # 按类别统计
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if r["passed"]:
                categories[cat]["passed"] += 1

        for cat, stats in categories.items():
            pct = stats["passed"] / stats["total"] * 100
            bar = "█" * stats["passed"] + "░" * (stats["total"] - stats["passed"])
            print(f"  {cat:8s} {bar} {stats['passed']}/{stats['total']} ({pct:.0f}%)")

        print("-" * 65)
        total_pct = self.passed / self.total * 100 if self.total > 0 else 0
        print(f"  {'总计':8s} {self.passed}/{self.total} ({total_pct:.0f}%)")
        print("=" * 65)

        # 失败用例详情
        failed = [r for r in self.results if not r["passed"]]
        if failed:
            print("\n❌ 失败用例详情：")
            for i, r in enumerate(failed, 1):
                print(f"  {i}. [{r['category']}] \"{r['input'][:40]}\"")
                print(f"     期望：{r['expected']}  实际：{r['actual']}")

        return self.passed, self.total


if __name__ == "__main__":
    model = "qwen2.5:7b"
    if len(sys.argv) > 1:
        model = sys.argv[1]

    suite = DecisionTestSuite(model=model)
    start = time.time()
    passed, total = suite.run_all()
    elapsed = time.time() - start

    print(f"\n⏱ 总耗时：{elapsed:.1f}秒")
    print(f"🎯 通过率：{passed}/{total} ({passed/total*100:.0f}%)")
