"""
分层决策示例 - 第四课配套
当选项超过 7 个时，用两级决策代替一级决策

一级决策（粗分类）：
  - tech_help → 技术帮助
  - content_process → 内容处理
  - unknown → 无法识别

二级决策（细分类）：
  tech_help 下：
    - answer_question → 回答问题
    - code_review → 代码审查
    - write_code → 编写代码

  content_process 下：
    - summarize_text → 文本摘要
    - translate → 翻译
"""

import sys
from agent import Agent


class HierarchicalRouter:
    """
    分层路由器

    当技能太多时，一级决策准确率会下降。
    分层决策先做大类判断，再做细分类，准确率更高。
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.agent = Agent(model=model)
        self.categories: dict[str, dict] = {}  # {大类: {子类: func}}

    def register(self, category: str, category_desc: str, skill_name: str, func: callable):
        """
        注册技能到分类树中

        Args:
            category: 大类名称（如 "tech_help"）
            category_desc: 大类描述（给 AI 看的）
            skill_name: 技能名称（如 "answer_question"）
            func: 技能函数
        """
        if category not in self.categories:
            self.categories[category] = {
                "description": category_desc,
                "skills": {},
            }
        self.categories[category]["skills"][skill_name] = func
        print(f"✅ {category} > {skill_name}")

    def route(self, user_input: str) -> dict:
        """
        两级路由

        第一步：判断大类
        第二步：在大类内判断具体技能
        """
        # 第一步：粗分类
        cat_choices = list(self.categories.keys())
        cat_choices.append("unknown")

        # 构建大类描述
        cat_details = []
        for cat in cat_choices:
            if cat == "unknown":
                cat_details.append("- unknown：无法判断用户意图")
            else:
                desc = self.categories[cat]["description"]
                cat_details.append(f"- {cat}：{desc}")

        cat_prompt = f"""根据用户输入，选择最合适的分类。

可选分类：
{chr(10).join(cat_details)}

规则：只返回 JSON，格式：{{"category": "分类名"}}
用户输入：{user_input}

请返回 JSON："""

        # 使用 Agent 的 decide 方法做第一级决策
        category = self.agent.decide(user_input, cat_choices)

        if category is None or category == "unknown":
            return {"success": False, "result": "🤔 无法识别你的意图", "category": None}

        if category not in self.categories:
            return {"success": False, "result": f"⚠️ 未知分类：{category}", "category": category}

        print(f"  📂 一级分类：{category}")

        # 第二步：细分类
        skills = self.categories[category]["skills"]
        skill_choices = list(skills.keys())

        # 如果该分类下只有一个技能，直接执行
        if len(skill_choices) == 1:
            skill_name = skill_choices[0]
            print(f"  🔧 二级选择：{skill_name}（唯一技能，自动选择）")
        else:
            # 用 decide 做第二级决策
            skill_name = self.agent.decide(user_input, skill_choices)
            print(f"  🔧 二级选择：{skill_name}")

        if skill_name and skill_name in skills:
            try:
                result = skills[skill_name](user_input)
                return {"success": True, "result": result, "category": category, "skill": skill_name}
            except Exception as e:
                return {"success": False, "result": f"⚠️ 执行出错：{e}", "category": category}

        return {"success": False, "result": "❌ 细分类失败", "category": category}

    def show_tree(self):
        """打印分类树"""
        print("\n" + "=" * 50)
        print("  🌳 技能分类树")
        print("=" * 50)
        for cat, info in self.categories.items():
            print(f"  📂 {cat} ({info['description']})")
            for skill_name in info["skills"]:
                print(f"     └─ 🔧 {skill_name}")
        print("=" * 50)


# ====== 示例使用 ======
if __name__ == "__main__":
    print("🌳 分层决策演示\n")

    router = HierarchicalRouter(model="qwen2.5:7b" if "--local" not in sys.argv else "qwen2.5:7b")

    # 定义技能
    def skill_answer(text, ctx=None):
        return f"📝 回答问题：{text}"

    def skill_review(text, ctx=None):
        return f"🔍 代码审查：{text}"

    def skill_write(text, ctx=None):
        return f"💻 编写代码：{text}"

    def skill_summarize(text, ctx=None):
        return f"📋 文本摘要：{text}"

    def skill_translate(text, ctx=None):
        return f"🌐 翻译：{text}"

    # 注册到分类树
    router.register("tech_help", "技术类问题（问答、代码相关）", "answer_question", skill_answer)
    router.register("tech_help", "技术类问题（问答、代码相关）", "code_review", skill_review)
    router.register("tech_help", "技术类问题（问答、代码相关）", "write_code", skill_write)
    router.register("content_process", "内容处理（摘要、翻译等）", "summarize_text", skill_summarize)
    router.register("content_process", "内容处理（摘要、翻译等）", "translate", skill_translate)

    router.show_tree()

    # 测试
    print("\n🧪 测试路由：\n")
    tests = [
        "什么是机器学习？",
        "帮我写一个爬虫",
        "总结一下这篇文章",
        "translate this to English",
    ]

    for test_input in tests:
        print(f"💬 用户：{test_input}")
        result = router.route(test_input)
        print(f"📤 结果：{result['result']}\n")
