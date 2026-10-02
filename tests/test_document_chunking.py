import io
import json
import unittest
import zipfile

from mcp.document_chunker import ChunkingConfig, DocumentChunker
from mcp.document_parser import parse_uploaded_document
from mcp.knowledge_base import KnowledgeBase
from mcp.lexical_index import LexicalRecord, SQLiteFTS5Index


class StructureAwareParserTests(unittest.TestCase):
    def test_markdown_preserves_headings_lists_tables_and_code(self):
        payload = """# 巡检方案
个人巡检方案说明。

- 每月能耗阈值
- 团队巡检权限

## 计费
| 项目 | 说明 |
|---|---|
| 能耗阈值 | 控制台为准 |

```text
401 AUTH_EXPIRED
```
""".encode("utf-8")

        document = parse_uploaded_document("support.md", payload)[0]
        types = [block["block_type"] for block in document["blocks"]]

        self.assertEqual(
            ["heading", "paragraph", "list", "heading", "table", "code"],
            types,
        )
        self.assertEqual("巡检方案 > 计费", document["blocks"][4]["heading_path"])

    def test_docx_preserves_heading_and_table_blocks(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>故障排查</w:t></w:r></w:p>
            <w:p><w:r><w:t>先记录控制器版本。</w:t></w:r></w:p>
            <w:tbl>
              <w:tr><w:tc><w:p><w:r><w:t>错误码</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>动作</w:t></w:r></w:p></w:tc></w:tr>
              <w:tr><w:tc><w:p><w:r><w:t>401</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>重新登录</w:t></w:r></w:p></w:tc></w:tr>
            </w:tbl>
          </w:body>
        </w:document>"""
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("word/document.xml", xml)

        document = parse_uploaded_document("排障.docx", payload.getvalue())[0]

        self.assertEqual(
            ["heading", "paragraph", "table"],
            [block["block_type"] for block in document["blocks"]],
        )
        self.assertEqual("故障排查", document["blocks"][2]["heading_path"])

    def test_json_upload_preserves_version_governance_metadata(self):
        payload = json.dumps([{
            "title": "团队巡检方案价格通知",
            "content": "团队巡检方案月费为 99 元。",
            "knowledge_key": "streetlight.plan.team_price",
            "fact_value": "99-cny",
            "knowledge_version": "2026-q3",
            "authority": "official",
            "effective_at": "2026-09-01T00:00:00Z",
            "reviewed_at": "2026-08-20T00:00:00Z",
            "freshness_ttl_days": 90,
            "scope": "team_plan",
            "audience": ["individual", "team"],
        }], ensure_ascii=False).encode("utf-8")

        document = parse_uploaded_document("streetlight-rules.json", payload)[0]

        self.assertEqual(
            "streetlight.plan.team_price", document["knowledge_key"]
        )
        self.assertEqual("99-cny", document["fact_value"])
        self.assertEqual(
            ["individual", "team"], document["audience"]
        )
        self.assertEqual("team_plan", document["scope"])


class TwoStageChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()

    def test_new_heading_is_a_hard_boundary(self):
        document = {
            "blocks": [
                {
                    "text": "巡检方案说明",
                    "block_type": "heading",
                    "heading_path": "巡检方案说明",
                    "block_index": 0,
                },
                {
                    "text": "巡检方案权益说明。" * 35,
                    "block_type": "paragraph",
                    "heading_path": "巡检方案说明",
                    "block_index": 1,
                },
                {
                    "text": "工单撤回说明",
                    "block_type": "heading",
                    "heading_path": "工单撤回说明",
                    "block_index": 2,
                },
                {
                    "text": "工单撤回资格以渠道规则为准。" * 25,
                    "block_type": "paragraph",
                    "heading_path": "工单撤回说明",
                    "block_index": 3,
                },
            ]
        }

        chunks = self.chunker.split(document)

        self.assertTrue(any(chunk.heading_path == "巡检方案说明" for chunk in chunks))
        self.assertTrue(any(chunk.heading_path == "工单撤回说明" for chunk in chunks))
        self.assertFalse(any(
            "巡检方案权益" in chunk.content and "工单撤回资格" in chunk.content
            for chunk in chunks
        ))

    def test_long_sentence_is_bounded_and_overlapped(self):
        chunks = self.chunker.split({"content": "无标点长句" * 260})

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk.content) <= 512 for chunk in chunks))
        self.assertTrue(any(
            chunk.overlap_from_previous > 0 for chunk in chunks[1:]
        ))

    def test_table_splits_by_rows_and_repeats_header(self):
        chunker = DocumentChunker(ChunkingConfig(
            target_chars=90,
            max_chars=120,
            min_chars=40,
            overlap_chars=0,
        ))
        rows = ["| 项目 | 说明 |", "|---|---|"] + [
            f"| 项目{index} | {'说明' * 15} |" for index in range(10)
        ]
        chunks = chunker.split({"blocks": [{
            "text": "\n".join(rows),
            "block_type": "table",
            "heading_path": "巡检方案",
            "block_index": 0,
        }]})

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.chunk_type == "table" for chunk in chunks))
        self.assertTrue(all(
            chunk.content.startswith("| 项目 | 说明 |\n|---|---|")
            for chunk in chunks
        ))

    def test_heading_is_attached_to_following_atomic_block(self):
        chunks = self.chunker.split({"blocks": [
            {
                "text": "错误码",
                "block_type": "heading",
                "heading_path": "错误码",
                "block_index": 0,
            },
            {
                "text": "| 错误码 | 动作 |\n|---|---|\n| 401 | 重新登录 |",
                "block_type": "table",
                "heading_path": "错误码",
                "block_index": 1,
            },
        ]})

        self.assertEqual(1, len(chunks))
        self.assertEqual("table", chunks[0].chunk_type)
        self.assertIn("错误码", chunks[0].content)


class KnowledgeBaseHybridTests(unittest.TestCase):
    class FakeCollection:
        def __init__(self):
            self.delete_calls = []
            self.upserts = []

        def delete(self, where):
            self.delete_calls.append(where)

        def upsert(self, ids, documents, metadatas):
            self.upserts.append({
                "ids": list(ids),
                "documents": list(documents),
                "metadatas": list(metadatas),
            })

        def get(self, ids=None, include=None):
            latest = self.upserts[-1]
            indexes = list(range(len(latest["ids"])))
            if ids is not None:
                wanted = set(ids)
                indexes = [
                    index for index in indexes
                    if latest["ids"][index] in wanted
                ]
            return {
                "ids": [latest["ids"][index] for index in indexes],
                "documents": [latest["documents"][index] for index in indexes],
                "metadatas": [latest["metadatas"][index] for index in indexes],
            }

    def setUp(self):
        self.kb = KnowledgeBase.__new__(KnowledgeBase)
        self.kb._chunker = DocumentChunker()
        self.kb._collection = self.FakeCollection()
        self.kb._lexical_index = SQLiteFTS5Index(":memory:")
        self.kb._embedding_function = None
        self.kb._rrf_k = 20.0
        self.kb._heading_lexical_weight = 0.50
        self.kb._vector_candidate_multiplier = 4
        self.kb._vector_candidate_min = 20
        self.document = {
            "title": "能耗阈值说明",
            "content": "能耗阈值异常时保留模型名称和页面截图。",
            "source_uri": "support.md",
            "parser_version": "coding-plan-structure-v2",
            "knowledge_key": "urbanops_streetlight.quota_troubleshooting",
            "fact_value": "collect-diagnostics",
            "knowledge_version": "2026.08",
            "authority": "official",
            "effective_at": "2026-08-01T00:00:00Z",
            "reviewed_at": "2026-08-20T00:00:00Z",
            "freshness_ttl_days": 90,
            "blocks": [{
                "text": "能耗阈值异常时保留模型名称和页面截图。",
                "block_type": "paragraph",
                "heading_path": "能耗阈值与用量",
                "page_number": 2,
                "block_index": 0,
                "source_start": 0,
                "source_end": 18,
            }],
        }

    def tearDown(self):
        self.kb._lexical_index.close()

    def test_reimport_uses_stable_ids_and_trace_metadata(self):
        first_count = self.kb.add_documents([self.document])
        first_ids = self.kb._collection.upserts[-1]["ids"]
        second_count = self.kb.add_documents([self.document])
        second_ids = self.kb._collection.upserts[-1]["ids"]
        metadata = self.kb._collection.upserts[-1]["metadatas"][0]

        self.assertEqual(1, first_count)
        self.assertEqual(first_count, second_count)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(2, len(self.kb._collection.delete_calls))
        self.assertEqual(2, metadata["page_start"])
        self.assertEqual("能耗阈值与用量", metadata["heading_path"])
        self.assertEqual("urbanops-structure-v1", metadata["splitter_version"])
        self.assertTrue(metadata["content_sha256"])
        self.assertEqual(
            "urbanops_streetlight.quota_troubleshooting", metadata["knowledge_key"]
        )
        self.assertEqual("2026.08", metadata["knowledge_version"])
        self.assertEqual("official", metadata["authority"])
        self.assertEqual(90, metadata["freshness_ttl_days"])

    def test_invalid_governance_metadata_rejects_batch_before_writes(self):
        invalid = {
            "title": "invalid mutable fact",
            "content": "This document has a value but no stable fact key.",
            "fact_value": "orphan-value",
        }

        with self.assertRaises(ValueError):
            self.kb.add_documents([self.document, invalid])

        self.assertEqual([], self.kb._collection.upserts)
        self.assertEqual([], self.kb._collection.delete_calls)

    def test_source_lookup_uses_stable_chunk_id(self):
        self.kb.add_documents([self.document])
        chunk_id = self.kb._collection.upserts[-1]["ids"][0]

        result = self.kb.source_lookup(chunk_id)

        self.assertEqual(1, len(result))
        self.assertEqual(chunk_id, result[0]["chunk_id"])
        self.assertEqual(2, result[0]["page_start"])
        self.assertEqual(
            "urbanops_streetlight.quota_troubleshooting", result[0]["knowledge_key"]
        )

    def test_chinese_fts_can_correct_misleading_vector_rank(self):
        class StaticCollection:
            def query(self, **kwargs):
                return {
                    "ids": [["login", "quota"]],
                    "documents": [[
                        "登录异常时检查系统时间。",
                        "能耗阈值异常时保留模型名称、时间和页面截图。",
                    ]],
                    "metadatas": [[
                        {
                            "document_id": "d1",
                            "chunk_id": "login",
                            "heading_path": "登录问题",
                            "chunk_index": 0,
                        },
                        {
                            "document_id": "d2",
                            "chunk_id": "quota",
                            "heading_path": "能耗阈值异常",
                            "chunk_index": 0,
                        },
                    ]],
                    "distances": [[0.1, 0.9]],
                }

        self.kb._collection = StaticCollection()
        self.kb._lexical_index.rebuild([
            LexicalRecord(
                "login",
                "d1",
                "登录异常时检查系统时间。",
                {
                    "document_id": "d1",
                    "chunk_id": "login",
                    "heading_path": "登录问题",
                },
            ),
            LexicalRecord(
                "quota",
                "d2",
                "能耗阈值异常时保留模型名称、时间和页面截图。",
                {
                    "document_id": "d2",
                    "chunk_id": "quota",
                    "heading_path": "能耗阈值异常",
                },
            ),
        ])

        result = self.kb.search("能耗阈值异常需要保留什么截图", top_k=2)

        self.assertEqual("quota", result[0]["chunk_id"])
        self.assertEqual("hybrid", result[0]["retrieval_mode"])
        self.assertGreater(
            result[0]["heading_lexical_score"],
            result[1]["heading_lexical_score"],
        )


class SQLiteLexicalIndexTests(unittest.TestCase):
    def test_replace_document_is_idempotent(self):
        index = SQLiteFTS5Index(":memory:")
        try:
            record = LexicalRecord(
                "quota",
                "d1",
                "能耗阈值异常需要保留页面截图。",
                {"title": "能耗阈值说明", "heading_path": "能耗阈值异常"},
            )
            index.replace_document("d1", [record])
            index.replace_document("d1", [record])

            self.assertEqual(1, index.count())
            self.assertEqual(
                ["quota"],
                [
                    item.chunk_id
                    for item in index.search(
                        "能耗阈值异常截图",
                        document_ids=["d1"],
                        limit=5,
                        heading_weight=0.5,
                    )
                ],
            )
        finally:
            index.close()


if __name__ == "__main__":
    unittest.main()
