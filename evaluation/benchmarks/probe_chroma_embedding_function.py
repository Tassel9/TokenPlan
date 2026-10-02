# -*- coding: utf-8 -*-
"""Probe: how does chromadb 0.5.23 validate a custom embedding_function?

Checks create/add/query, reopening the same collection with the same function,
and reopening with a different function (dimension/profile mismatch).
"""
from __future__ import annotations

import pathlib
import shutil
import tempfile

import chromadb


class DummyEF:
    def __init__(self, dim=8, tag="dummy"):
        self.dim = dim
        self.tag = tag

    def __call__(self, input):  # chroma passes a list of documents/queries
        return [[float(len(text) + index) / 10.0 for index in range(self.dim)] for text in input]

    def name(self):
        return f"dummy-{self.tag}"


def main() -> int:
    root = pathlib.Path(tempfile.mkdtemp(prefix="chroma-ef-probe-"))
    client = chromadb.PersistentClient(path=str(root), settings=chromadb.Settings(anonymized_telemetry=False))

    print("1) create with custom EF")
    collection = client.get_or_create_collection(name="probe_v1", embedding_function=DummyEF())
    collection.add(ids=["a", "b"], documents=["工单撤回规则说明", "网络认证失败排查"], metadatas=[{"document_id": "a"}, {"document_id": "b"}])
    print("   count:", collection.count())
    result = collection.query(query_texts=["工单撤回"], n_results=2)
    print("   query ids:", result["ids"])
    print("   distance:", result["distances"])

    print("2) reopen same collection + same EF (new process semantics)")
    client2 = chromadb.PersistentClient(path=str(root), settings=chromadb.Settings(anonymized_telemetry=False))
    try:
        reopened = client2.get_or_create_collection(name="probe_v1", embedding_function=DummyEF())
        print("   ok, count:", reopened.count(), "query:", reopened.query(query_texts=["工单撤回"], n_results=1)["ids"])
    except Exception as ex:
        print("   FAILED:", type(ex).__name__, ex)

    print("3) reopen same collection + different EF dim")
    try:
        other = client2.get_or_create_collection(name="probe_v1", embedding_function=DummyEF(dim=16, tag="other"))
        print("   no error; query:", other.query(query_texts=["工单撤回"], n_results=1)["ids"])
    except Exception as ex:
        print("   raised:", type(ex).__name__, str(ex)[:200])

    print("4) custom EF object without name()/config on a fresh collection")
    class BareEF:
        def __call__(self, input):
            return [[0.1] * 4 for _ in input]

    try:
        bare = client2.get_or_create_collection(name="probe_v2", embedding_function=BareEF())
        bare.add(ids=["x"], documents=["测试"])
        print("   ok, count:", bare.count())
    except Exception as ex:
        print("   raised:", type(ex).__name__, str(ex)[:300])

    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
