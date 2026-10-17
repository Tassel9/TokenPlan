"""Keep the restored recall parameters aligned with the frozen policy fixture.

The fixture's historical quality measurements are not current quality evidence.
"""
import json
import unittest
from pathlib import Path

from memory.conversation_memory import MEMORY_KEY_LABELS, MemoryManager


DATASET = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "datasets"
    / "long_term_memory_recall_v1.json"
)


class LongTermMemoryRecallCalibrationTests(unittest.TestCase):
    def test_fixed_dataset_preserves_historical_recall_policy(self):
        payload = json.loads(DATASET.read_text(encoding="utf-8"))

        self.assertEqual("long-term-memory-recall-v1", payload["schema_version"])
        self.assertRegex(payload["embedding_revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(0.58, payload["distance_threshold"])
        self.assertEqual(3, payload["max_recalled_facts"])
        self.assertEqual(payload["distance_threshold"], MemoryManager.FACT_MAX_DISTANCE)
        self.assertEqual(payload["max_recalled_facts"], MemoryManager.FACT_RECALL_MAX)
        self.assertEqual(
            set(MEMORY_KEY_LABELS),
            {fact["memory_key"] for fact in payload["facts"]},
        )
        self.assertEqual(
            len(payload["cases"]),
            len({case["case_id"] for case in payload["cases"]}),
        )


if __name__ == "__main__":
    unittest.main()
