"""Client for interacting with a local Ollama model."""

from __future__ import annotations

import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class OllamaClient:
    """Lightweight wrapper around the Ollama HTTP API."""

    def __init__(self, model: str = "gemma:2b", timeout: int = 60) -> None:
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=self.timeout)
        logger.info("OllamaClient initialized with model %s", self.model)

    async def generate(self, prompt: str) -> str:
        """Send ``prompt`` to Ollama and return the generated text."""

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = await self._client.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    async def answer_question(self, context: str, question: str) -> str:
        """Build an optimized prompt with ``context`` and return the answer."""

        prompt = (
            "You are an assistant answering questions based on the given context.\n"
            "Respond concisely. If the answer cannot be found in the context,"
            " reply with 'Answer not found'.\n\nContext:\n"
            f"{context}\n\nQuestion:\n{question}\nAnswer:"
        )
        return await self.generate(prompt)

    async def extract_entities(self, query: str) -> str:
        """Extract structured entities from ``query`` as JSON."""

        prompt = (
            "Extract key details such as age, procedure, location and policy duration "
            "from the following query. Return a JSON object with these fields if present.\nQuery:\n"
            f"{query}\nJSON:"
        )
        return await self.generate(prompt)

    async def rag_answer(self, question: str, context: str) -> str:
        """Answer ``question`` using ``context`` and return JSON output."""

        prompt = (
            "You are an insurance assistant. Use only the provided context to answer the question. "
            "Respond in JSON with fields: decision, amount, justification.\n\nContext:\n"
            f"{context}\n\nQuestion:\n{question}\nAnswer JSON:"
        )
        return await self.generate(prompt)

    async def test_connection(self) -> bool:
        """Basic check to see if the Ollama service is reachable."""

        try:
            resp = await self.answer_question("test", "reply with ok")
            return bool(resp)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error("Ollama connection test failed: %s", exc)
            return False

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""

        await self._client.aclose()

