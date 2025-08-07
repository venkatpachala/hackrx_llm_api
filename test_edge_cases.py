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
        # Always return stored entries with score 1.0
        return [{**e, "score": 1.0} for e in self.entries[:k]]


async def fake_extract(self, query: str) -> str:
    return '{"procedure": "angioplasty"}'


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("utils.vector_store.VectorStore", DummyVectorStore)
    monkeypatch.setattr("main.VectorStore", DummyVectorStore)
    monkeypatch.setattr("utils.ollama_client.OllamaClient.extract_entities", fake_extract)
    from utils.document_loader import Chunk

    async def fake_process_bytes(self, data: bytes, name: str):
        text = data.decode()
        # text deliberately does not mention angioplasty
        return [Chunk(chunk_id=0, file_name=name, page_range="1", text=text, section="inclusion")]

    monkeypatch.setattr("utils.document_loader.DocumentLoader.process_bytes", fake_process_bytes)

    async def fake_rag(self, question: str, clauses: list[dict], edge_instruction: str = "") -> str:
        raise AssertionError("rag_answer should not be called when procedure missing")

    monkeypatch.setattr("utils.ollama_client.OllamaClient.rag_answer", fake_rag)


def test_procedure_not_found():
    files = [
        (
            "file",
            ("doc1.txt", b"General hospitalization is covered.", "text/plain"),
        )
    ]
    data = {"question": "56-year-old male wants angioplasty"}
    headers = {"Authorization": "Bearer testtoken"}

    resp = client.post("/hackrx/run", files=files, data=data, headers=headers)
    assert resp.status_code == 200
    ans = resp.json()["answers"][0]
    assert ans["decision"] == "insufficient info"
    assert "do not explicitly mention angioplasty" in ans["justification"].lower()
