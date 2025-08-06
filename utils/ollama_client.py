"""Client for interacting with a local Ollama model."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests


logger = logging.getLogger(__name__)


class OllamaClient:
    """Lightweight wrapper around the Ollama HTTP API."""

    def __init__(self, model: str = "gemma:2b", timeout: int = 60) -> None:
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        self.timeout = timeout
        logger.info("OllamaClient initialized with model %s", self.model)

    def _post(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    async def generate(self, prompt: str) -> str:
        """Execute the blocking HTTP request in a thread."""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._post(prompt))

    async def answer_question(self, context: str, question: str) -> str:
        """Build an optimized prompt with ``context`` and return the answer."""

        prompt = (
            "You are an assistant answering questions based on the given context.\n"
            "Respond concisely. If the answer cannot be found in the context,"
            " reply with 'Answer not found'.\n\nContext:\n"
            f"{context}\n\nQuestion:\n{question}\nAnswer:"
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

