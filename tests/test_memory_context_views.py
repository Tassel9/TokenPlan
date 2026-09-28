import unittest

from agents.specialist_agents import AgentInput, _execution_context
from memory.conversation_memory import (
    LongTermMemoryContext,
    Message,
    MsgRole,
    ShortTermMemoryContext,
)


class MemoryContextViewTests(unittest.TestCase):
    def test_short_term_memory_has_no_long_term_or_working_state(self):
        memory = ShortTermMemoryContext(
            recent_messages=[
                Message(MsgRole.USER, "插件登录失败"),
                Message(MsgRole.ASSISTANT, "请提供错误码"),
            ],
            summary="正在排查插件登录问题",
        )

        context = memory.to_text()

        self.assertIn("最近对话", context)
        self.assertIn("短期会话摘要", context)
        self.assertLess(context.index("短期会话摘要"), context.index("最近对话"))
        self.assertFalse(hasattr(memory, "current_profile"))
        self.assertFalse(hasattr(memory, "case_state"))

    def test_long_term_memory_contains_only_current_profile_and_current_facts(self):
        memory = LongTermMemoryContext(
            current_profile={
                "communication_style": ["先给结论"],
                "facts": {"style.answer_order": "先给结论"},
            },
            recalled_facts=["environment.os=Windows"],
        )

        context = memory.to_text()

        self.assertIn("当前有效用户特征", context)
        self.assertIn("先给结论", context)
        self.assertIn("相关用户事实", context)
        self.assertIn("environment.os=Windows", context)
        self.assertNotIn("历史", context)
        self.assertFalse(hasattr(memory, "recent_messages"))
        self.assertFalse(hasattr(memory, "case_state"))

    def test_runtime_adapter_labels_memory_sources_without_changing_ownership(self):
        req = AgentInput(
            request_id="request-1",
            message="继续",
            execution_query="继续",
            user_id="user-1",
            conv_id="conv-1",
            intent_id="intent-1",
            intent="facility_troubleshooting",
            short_term_context="正在排查插件登录问题",
            long_term_context="用户偏好先给结论",
            case_state={"pending_slots": ["error_code"]},
        )

        context = _execution_context(req)

        self.assertIn("工作记忆：当前任务/业务状态", context)
        self.assertIn("[短期记忆]", context)
        self.assertIn("[长期记忆]", context)
        self.assertLess(context.index("[短期记忆]"), context.index("[长期记忆]"))


if __name__ == "__main__":
    unittest.main()
