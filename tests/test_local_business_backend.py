import tempfile
import unittest
from pathlib import Path

from mcp.local_business_backend import UrbanOpsLocalBackend
from mcp.tool_capabilities import BUSINESS_DATA_QUERY, BUSINESS_OPERATION_EXECUTE
from mcp.tool_registry import Tool, ToolRegistry


class UrbanOpsLocalBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = UrbanOpsLocalBackend(
            str(Path(self.temp_dir.name) / "urbanops_local.sqlite3")
        )

    def tearDown(self):
        self.service.close()
        self.temp_dir.cleanup()

    async def test_query_is_user_scoped_and_parameterized(self):
        self.service.upsert_facility(
            "FAC-1",
            "user-1",
            asset_type="排水泵站",
            status="online",
            location="东城区",
        )
        self.service.upsert_work_order(
            "WO-1",
            "user-1",
            facility_id="FAC-1",
            status="processing",
            priority="high",
        )

        facility = await self.service.query(
            {"resource": "facility", "record_id": "FAC-1"},
            {"user_id": "user-1"},
        )
        own_work_order = await self.service.query(
            {"resource": "work_order", "record_id": "WO-1"},
            {"user_id": "user-1"},
        )
        other_user = await self.service.query(
            {"resource": "work_order", "record_id": "WO-1"},
            {"user_id": "user-2"},
        )

        self.assertEqual("排水泵站", facility.data["record"]["asset_type"])
        self.assertEqual("processing", own_work_order.data["record"]["status"])
        self.assertFalse(other_user.data["found"])
        self.assertIsNone(other_user.data["record"])

    async def test_operation_is_approved_and_idempotent(self):
        context = {
            "user_id": "user-1",
            "approval_id": "approval-1",
            "idempotency_key": "idem-1",
        }
        params = {
            "operation": "withdraw_work_order",
            "target_id": "WO-1",
            "details": {"reason": "duplicate report"},
        }

        first = await self.service.submit_operation(params, context)
        replay = await self.service.submit_operation(params, context)

        self.assertFalse(first.data["replayed"])
        self.assertTrue(replay.data["replayed"])
        self.assertEqual(first.data["request_id"], replay.data["request_id"])
        self.assertEqual("request_accepted_only", first.data["completion_claim"])
        with self.assertRaisesRegex(ValueError, "different operation"):
            await self.service.submit_operation(
                {"operation": "assign_work_order", "target_id": "WO-1"},
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
            {"operation": "withdraw_work_order", "target_id": "WO-1"},
            context={
                "user_id": "user-1",
                "agent_type": "business_data_query",
            },
        )
        missing_approval = await registry.call(
            "business_operation",
            {"operation": "withdraw_work_order", "target_id": "WO-1"},
            context={
                "user_id": "user-1",
                "agent_type": "business_operation",
            },
        )
        accepted = await registry.call(
            "business_operation",
            {"operation": "withdraw_work_order", "target_id": "WO-1"},
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
