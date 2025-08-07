import os
import pytest
from fastapi.testclient import TestClient

from main import app

os.environ["API_TOKEN"] = "testtoken"
client = TestClient(app)


class DummyVectorStore:
    def __init__(self):
        self.entries = []

    async def add_texts(self, texts, metadatas):
        self.entries.extend([{**m, "text": t} for t, m in zip(texts, metadatas)])

    async def similarity_search(self, query: str, k: int = 5, section: str | None = None):
        res = []
        for e in self.entries:
            if section and e.get("section") != section:
                continue
            item = {**e, "score": 1.0}
            res.append(item)
            if len(res) >= k:
                break
        return res


async def fake_extract(self, query: str) -> str:
    return '{"procedure": "knee surgery"}'


async def fake_rag(self, question: str, clauses: list[dict], edge_instruction: str = "") -> str:
    return '{"decision": "approved", "amount": "50000", "justification": "Clause 12.3, Page 1"}'


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("utils.vector_store.VectorStore", DummyVectorStore)
    monkeypatch.setattr("main.VectorStore", DummyVectorStore)
    monkeypatch.setattr("utils.ollama_client.OllamaClient.extract_entities", fake_extract)
    monkeypatch.setattr("utils.ollama_client.OllamaClient.rag_answer", fake_rag)
    from utils.document_loader import Chunk

    async def fake_process_bytes(self, data: bytes, name: str):
        text = data.decode()
        return [Chunk(chunk_id=0, file_name=name, page_range="1", text=text, section="inclusion")]

    monkeypatch.setattr("utils.document_loader.DocumentLoader.process_bytes", fake_process_bytes)


def test_multi_document_query():
    files = [
        (
            "file",
            ("doc1.txt", b"Clause 12.3, Page 1: Knee surgery is covered.", "text/plain"),
        ),
        (
            "file",
            (
                "doc2.txt",
                b"Clause 8.1, Page 5: For knee surgery, maximum amount is 50000 INR.",
                "text/plain",
            ),
        ),
        (
            "file",
            ("doc3.txt", b"Clause 7.3, Page 9: Heart surgery details.", "text/plain"),
        ),
    ]
    data = {"question": "46-year-old male, knee surgery in Pune, 3-month-old policy"}
    headers = {"Authorization": "Bearer testtoken"}

    resp = client.post("/hackrx/run", files=files, data=data, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["answers"][0]["decision"] == "approved"
    assert body["answers"][0]["amount"] == "50000"
    assert body["answers"][0]["query"] == data["question"]
    assert body["answers"][0]["relevant_clauses"][0]["file"] == "doc1.txt"
    assert body["answers"][0]["relevant_clauses"][0]["page"] == "1"
    assert "Clause" in body["answers"][0]["justification"]
