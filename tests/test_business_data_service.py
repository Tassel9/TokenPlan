import tempfile
import unittest
from pathlib import Path

from mcp.business_data_service import BusinessDataService
from mcp.tool_capabilities import BUSINESS_DATA_QUERY, BUSINESS_OPERATION_EXECUTE
from mcp.tool_registry import Tool, ToolRegistry


class BusinessDataServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = BusinessDataService(
            str(Path(self.temp_dir.name) / "business.sqlite3")
        )

    def tearDown(self):
        self.service.close()
        self.temp_dir.cleanup()

    async def test_query_is_user_scoped_and_parameterized(self):
        self.service.upsert_account(
            "user-1",
            plan="pro",
            subscription_status="active",
            quota_remaining=1200,
        )
        self.service.upsert_order(
            "ORDER-1",
            "user-1",
            order_type="subscription",
            status="paid",
            amount=99,
        )

        account = await self.service.query(
            {"resource": "account"}, {"user_id": "user-1"}
        )
        own_order = await self.service.query(
            {"resource": "order", "record_id": "ORDER-1"},
            {"user_id": "user-1"},
        )
        other_user = await self.service.query(
            {"resource": "order", "record_id": "ORDER-1"},
            {"user_id": "user-2"},
        )

        self.assertEqual("pro", account.data["record"]["plan"])
        self.assertEqual("paid", own_order.data["record"]["status"])
        self.assertFalse(other_user.data["found"])
        self.assertIsNone(other_user.data["record"])

    async def test_operation_is_approved_and_idempotent(self):
        context = {
            "user_id": "user-1",
            "approval_id": "approval-1",
            "idempotency_key": "idem-1",
        }
        params = {
            "operation": "request_refund",
            "target_id": "ORDER-1",
            "details": {"reason": "duplicate charge"},
        }

        first = await self.service.submit_operation(params, context)
        replay = await self.service.submit_operation(params, context)

        self.assertFalse(first.data["replayed"])
        self.assertTrue(replay.data["replayed"])
        self.assertEqual(first.data["request_id"], replay.data["request_id"])
        self.assertEqual("request_accepted_only", first.data["completion_claim"])
        with self.assertRaisesRegex(ValueError, "different operation"):
            await self.service.submit_operation(
                {"operation": "request_invoice", "target_id": "ORDER-1"},
                context,
            )

    async def test_tool_permissions_separate_query_and_operation_agents(self):
        registry = ToolRegistry()
        registry.register(Tool(
            name="business_data_query",
            description="query",
            handler=self.service.query,
            schema={
                "type": "object",
                "properties": {"resource": {"type": "string"}},
                "required": ["resource"],
            },
            allowed_agents=["business_data_query"],
            capabilities=[BUSINESS_DATA_QUERY],
        ))
        registry.register(Tool(
            name="business_operation",
            description="operate",
            handler=self.service.submit_operation,
            schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "target_id": {"type": "string"},
                },
                "required": ["operation"],
            },
            side_effect="write",
            risk_level="high",
            allowed_agents=["business_operation"],
            capabilities=[BUSINESS_OPERATION_EXECUTE],
        ))

        denied = await registry.call(
            "business_operation",
            {"operation": "request_refund", "target_id": "ORDER-1"},
            context={
                "user_id": "user-1",
                "agent_type": "business_data_query",
            },
        )
        missing_approval = await registry.call(
            "business_operation",
            {"operation": "request_refund", "target_id": "ORDER-1"},
            context={
                "user_id": "user-1",
                "agent_type": "business_operation",
            },
        )
        accepted = await registry.call(
            "business_operation",
            {"operation": "request_refund", "target_id": "ORDER-1"},
            context={
                "user_id": "user-1",
                "agent_type": "business_operation",
                "approval_id": "approval-2",
                "idempotency_key": "idem-2",
            },
        )

        self.assertFalse(denied.success)
        self.assertIn("无权", denied.error)
        self.assertFalse(missing_approval.success)
        self.assertIn("approval_id", missing_approval.error)
        self.assertTrue(accepted.success)
        self.assertEqual("accepted", accepted.data["status"])


if __name__ == "__main__":
    unittest.main()
