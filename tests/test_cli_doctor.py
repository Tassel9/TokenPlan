"""Contract tests for ``python -m cli doctor`` and the CLI dispatch."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cli
from core import doctor


def _resolve(environ):
    """Run checks with a stub probe that reports every port as reachable."""

    def probe(host, port, timeout_s):
        return True, f"{host}:{port} 可连接"

    return doctor.run_checks(environ, probe=probe, timeout_s=0.01)


def _stub_probe(*unreachable):
    """Fail exactly the given hosts or (host, port) pairs; everything else is up."""

    entries = [(item, None) if isinstance(item, str) else (item[0], item[1]) for item in unreachable]

    def probe(host, port, timeout_s):
        for want_host, want_port in entries:
            if host == want_host and (want_port is None or want_port == port):
                return False, f"{host}:{port} 不可连接（ConnectionRefusedError）"
        return True, f"{host}:{port} 可连接"

    return probe


BASE_ENV = {
    "DEEPSEEK_API_KEY": "test-api-key-placeholder",
    "SESSION_DB_PATH": "./data/session/conversations.sqlite3",
    "CHROMA_HOST": "localhost",
    "CHROMA_PORT": "8001",
    "RABBITMQ_URL": "amqp://tokenplan:tokenplan123@localhost:5672/",
}


class DoctorCheckTests(unittest.TestCase):
    def test_missing_api_key_blocks_startup(self):
        checks = doctor.run_checks({}, probe=_stub_probe(), timeout_s=0.01)
        key_check = next(c for c in checks if c.key == "deepseek_api_key")
        self.assertEqual(key_check.status, doctor.FAIL)
        self.assertEqual(doctor.verdict(checks), "blocked")

    def test_placeholder_api_key_is_treated_as_unset(self):
        env = dict(BASE_ENV, DEEPSEEK_API_KEY=doctor.PLACEHOLDER_KEY)
        checks = doctor.run_checks(env, probe=_stub_probe(), timeout_s=0.01)
        key_check = next(c for c in checks if c.key == "deepseek_api_key")
        self.assertEqual(key_check.status, doctor.FAIL)
        self.assertIn("占位值", key_check.detail)

    def test_unwritable_sqlite_path_blocks_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent_file = Path(tmp) / "file"
            parent_file.write_text("occupied", encoding="utf-8")
            env = dict(BASE_ENV, SESSION_DB_PATH=str(parent_file / "sessions.sqlite3"))
            checks = doctor.run_checks(env, probe=_stub_probe(), timeout_s=0.01)
        storage = next(c for c in checks if c.key == "sqlite_session")
        self.assertEqual(storage.status, doctor.FAIL)
        self.assertIn("SESSION_DB_PATH", storage.hint)
        self.assertEqual(doctor.verdict(checks), "blocked")

    def test_chroma_down_is_fatal_without_embedded_fallback(self):
        checks = doctor.run_checks(
            BASE_ENV, probe=_stub_probe(("localhost", 8001)), timeout_s=0.01
        )
        chroma_check = next(c for c in checks if c.key == "chromadb")
        self.assertEqual(chroma_check.status, doctor.FAIL)

    def test_chroma_down_degrades_when_embedded_fallback_enabled(self):
        env = dict(BASE_ENV, MEMORY_ALLOW_EMBEDDED_CHROMA_FALLBACK="true")
        checks = doctor.run_checks(env, probe=_stub_probe(("localhost", 8001)), timeout_s=0.01)
        chroma_check = next(c for c in checks if c.key == "chromadb")
        self.assertEqual(chroma_check.status, doctor.WARN)
        self.assertIn("降级可用", chroma_check.detail)

    def test_rabbitmq_down_is_fatal_while_queue_enabled(self):
        checks = doctor.run_checks(
            BASE_ENV, probe=_stub_probe(("localhost", 5672)), timeout_s=0.01
        )
        mq_check = next(c for c in checks if c.key == "rabbitmq")
        self.assertEqual(mq_check.status, doctor.FAIL)
        self.assertEqual(doctor.verdict(checks), "blocked")

    def test_rabbitmq_down_is_degradable_when_queue_disabled(self):
        env = dict(BASE_ENV, LONG_TERM_MEMORY_QUEUE_ENABLED="false")
        checks = doctor.run_checks(env, probe=_stub_probe(("localhost", 5672)), timeout_s=0.01)
        mq_check = next(c for c in checks if c.key == "rabbitmq")
        self.assertEqual(mq_check.status, doctor.WARN)
        self.assertEqual(doctor.summarize(checks)["fail"], 0)
        self.assertEqual(doctor.verdict(checks), "degraded")
        self.assertTrue(any(c.key == "long_term_memory_queue" for c in checks))

    def test_trace_fingerprint_key_costs_a_warning_only(self):
        checks = doctor.run_checks(
            BASE_ENV, probe=_stub_probe(), timeout_s=0.01
        )
        fingerprint = next(c for c in checks if c.key == "trace_fingerprint_key")
        self.assertEqual(fingerprint.status, doctor.WARN)
        self.assertEqual(doctor.verdict(checks), "degraded")

    def test_healthy_configuration_reports_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog"
            catalog.mkdir()
            (catalog / "billing.yaml").write_text("id: billing\n", encoding="utf-8")
            trace_dir = Path(tmp) / "data"
            env = dict(
                BASE_ENV,
                SKILL_CATALOG_PATH=str(catalog),
                TRACE_DB_PATH=str(trace_dir / "trace.sqlite3"),
                TRACE_FINGERPRINT_KEY="unit-test-salt",
            )
            checks = doctor.run_checks(env, probe=_stub_probe(), timeout_s=0.01)

        self.assertEqual(doctor.verdict(checks), "ok")
        self.assertEqual(doctor.summarize(checks)["fail"], 0)
        text = doctor.render_text(checks, "ok")
        self.assertIn("可以直接启动", text)

    def test_missing_skill_catalog_is_fatal(self):
        env = dict(BASE_ENV, SKILL_CATALOG_PATH=str(Path(tempfile.gettempdir()) / "no-such-catalog"))
        checks = doctor.run_checks(env, probe=_stub_probe(), timeout_s=0.01)
        catalog_check = next(c for c in checks if c.key == "skill_catalog")
        self.assertEqual(catalog_check.status, doctor.FAIL)

    def test_json_payload_is_stable(self):
        checks = doctor.run_checks(BASE_ENV, probe=_stub_probe(), timeout_s=0.01)
        payload = doctor.to_payload(checks, doctor.verdict(checks))
        self.assertEqual(set(payload), {"verdict", "summary", "checks"})
        self.assertEqual(set(payload["summary"]), {"ok", "warn", "fail"})
        self.assertTrue(all(set(c) == {"key", "status", "detail", "hint"} for c in payload["checks"]))
        json.dumps(payload, ensure_ascii=False)

    def test_endpoint_parser_handles_urls_and_bare_hosts(self):
        self.assertEqual(doctor.parse_endpoint("amqp://tokenplan:pw@localhost:5672/", 5672), ("localhost", 5672))
        self.assertEqual(doctor.parse_endpoint("localhost", 8000), ("localhost", 8000))
        self.assertIsNone(doctor.parse_endpoint("", 8000))


class CliDispatchTests(unittest.TestCase):
    def test_split_command_routes_doctor(self):
        self.assertEqual(cli.split_command(["doctor", "--json"]), ("doctor", ["--json"]))
        self.assertEqual(cli.split_command(["--doctor"]), ("doctor", []))
        self.assertEqual(cli.split_command(["你好"]), ("chat", ["你好"]))
        self.assertEqual(cli.split_command([]), ("chat", []))

    def test_doctor_exit_code_is_zero_when_degraded(self):
        buffer = io.StringIO()
        with patch.object(cli, "run_checks", return_value=doctor.run_checks(
            BASE_ENV, probe=_stub_probe(), timeout_s=0.01
        )):
            with redirect_stdout(buffer):
                code = cli.main(["doctor"])
        self.assertEqual(code, 0)
        self.assertIn("TokenPlan 配置自检", buffer.getvalue())

    def test_doctor_exit_code_is_one_when_blocked(self):
        buffer = io.StringIO()
        with patch.object(cli, "run_checks", return_value=doctor.run_checks(
            {}, probe=_stub_probe(), timeout_s=0.01
        )):
            with redirect_stdout(buffer):
                code = cli.main(["doctor", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["verdict"], "blocked")

    def test_doctor_never_builds_the_service_graph(self):
        """Doctor must stay usable before Chroma/MQ are up."""

        with patch.object(cli, "run_once", side_effect=AssertionError("doctor must not chat")):
            with redirect_stdout(io.StringIO()), patch.object(
                cli, "run_checks", return_value=doctor.run_checks(
                    BASE_ENV, probe=_stub_probe(), timeout_s=0.01
                )
            ):
                self.assertEqual(cli.main(["doctor"]), 0)

    def test_service_graph_is_imported_lazily(self):
        """A bare environment (no Chroma/MQ drivers) must still self-check."""

        source = (Path(cli.__file__)).read_text(encoding="utf-8")
        header = source.split("async def run_once")[0]
        self.assertNotIn("import build_app_services", header)
        self.assertIn("from core.doctor import", header)


if __name__ == "__main__":
    unittest.main()
