"""Load small, reviewed knowledge packs bundled with the application."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "urbanops-knowledge-pack-v1"
DEFAULT_PACK_DIRECTORY = pathlib.Path(__file__).with_name("knowledge_packs")


class KnowledgePackError(ValueError):
    """Raised when a bundled knowledge pack violates the local contract."""


def load_packaged_knowledge(
    directory: Optional[pathlib.Path] = None,
) -> List[Dict[str, Any]]:
    """Return validated documents from every bundled JSON pack.

    The loader is deliberately independent from Chroma and FTS5. It only owns
    the on-disk content contract, which keeps ingestion and retrieval unchanged.
    """
    root = pathlib.Path(directory or DEFAULT_PACK_DIRECTORY)
    if not root.exists():
        return []

    documents: List[Dict[str, Any]] = []
    document_ids = set()
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
            raise KnowledgePackError(f"无法读取知识包 {path.name}: {ex}") from ex
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise KnowledgePackError(f"知识包 {path.name} 的 schema_version 不受支持")
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise KnowledgePackError(f"知识包 {path.name} 必须包含非空 documents")

        pack_id = str(payload.get("pack_id") or path.stem).strip()
        pack_version = str(payload.get("pack_version") or "").strip()
        for index, raw_document in enumerate(raw_documents):
            if not isinstance(raw_document, dict):
                raise KnowledgePackError(
                    f"知识包 {path.name} documents[{index}] 必须是对象"
                )
            document = dict(raw_document)
            for field in ("document_id", "title", "content", "source_uri"):
                if not str(document.get(field) or "").strip():
                    raise KnowledgePackError(
                        f"知识包 {path.name} documents[{index}].{field} 不能为空"
                    )
            document_id = str(document["document_id"]).strip()
            if document_id in document_ids:
                raise KnowledgePackError(f"知识包文档 ID 重复: {document_id}")
            document_ids.add(document_id)
            document.setdefault("knowledge_version", pack_version)
            document["pack_id"] = pack_id
            documents.append(document)
    return documents
