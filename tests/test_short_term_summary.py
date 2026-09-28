import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from memory.short_term_summary import (
    ShortTermSummaryV2,
    parse_summary_tool_response,
)


class ShortTermSummaryContractTests(unittest.TestCase):
    @staticmethod
    def _payload():
        return {
            "schema_version": "short-term-summary-v2",
            "current_goal": {
                "text": "排查登录失败",
                "source_turn_seqs": [7],
            },
            "confirmed_information": [{
                "text": "用户已经重启客户端",
                "source_turn_seqs": [7],
            }],
            "open_questions": [{
                "text": "仍需提供错误码",
                "source_turn_seqs": [7],
            }],
        }

    def test_complete_payload_validates_and_renders(self):
        summary = ShortTermSummaryV2.model_validate(self._payload())

        self.assertEqual({7}, summary.source_turn_seqs())
        self.assertIn("排查登录失败", summary.to_context_text())
        self.assertIn("仍需提供错误码", summary.to_context_text())

    def test_missing_required_field_fails_closed(self):
        payload = self._payload()
        payload.pop("current_goal")

        with self.assertRaises(ValidationError):
            ShortTermSummaryV2.model_validate(payload)

    def test_unknown_field_fails_closed(self):
        payload = self._payload()
        payload["unexpected"] = "value"

        with self.assertRaises(ValidationError):
            ShortTermSummaryV2.model_validate(payload)

    def test_duplicate_items_fail_closed(self):
        payload = self._payload()
        payload["confirmed_information"].append({
            "text": "用户已经重启客户端",
            "source_turn_seqs": [8],
        })

        with self.assertRaises(ValidationError):
            ShortTermSummaryV2.model_validate(payload)

    def test_plain_text_response_is_not_accepted(self):
        response = SimpleNamespace(content="当前目标：排查登录")

        with self.assertRaisesRegex(ValueError, "Tool Call"):
            parse_summary_tool_response(response)

    def test_exactly_one_summary_tool_call_is_required(self):
        block = SimpleNamespace(
            type="tool_use",
            name="submit_short_term_summary",
            input=self._payload(),
        )
        response = SimpleNamespace(content=[block, block])

        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_summary_tool_response(response)


if __name__ == "__main__":
    unittest.main()
