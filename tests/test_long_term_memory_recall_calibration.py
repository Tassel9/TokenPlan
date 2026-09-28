import json
import unittest
from pathlib import Path

from core.embedding_provider import BGE_DEFAULT_REVISION
from memory.conversation_memory import MEMORY_KEY_LABELS, MemoryManager


DATASET = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "datasets"
    / "long_term_memory_recall_v1.json"
)


class LongTermMemoryRecallCalibrationTests(unittest.TestCase):
    def test_fixed_dataset_matches_runtime_recall_policy(self):
        payload = json.loads(DATASET.read_text(encoding="utf-8"))

        self.assertEqual("long-term-memory-recall-v1", payload["schema_version"])
        self.assertEqual(BGE_DEFAULT_REVISION, payload["embedding_revision"])
        self.assertEqual(MemoryManager.FACT_MAX_DISTANCE, payload["distance_threshold"])
        self.assertEqual(MemoryManager.FACT_RECALL_MAX, payload["max_recalled_facts"])
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
