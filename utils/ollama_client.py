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

    async def rag_answer(
        self, question: str, clauses: list[dict], edge_instruction: str = ""
    ) -> str:
        """Answer ``question`` using retrieved ``clauses`` and return JSON."""

        lines = []
        for idx, c in enumerate(clauses, start=1):
            lines.append(
                f"{idx}. From {c.get('file_name', '')}, page {c.get('page_range', '')}:\n   \"{c.get('text', '')}\""
            )
        clause_block = "\n\n".join(lines) if lines else "None provided."

        prompt = (
            "You are a legal/insurance assistant.\n\n"
            f"User Query: \"{question}\"\n\n"
            "Here are the clauses retrieved from the documents:\n\n"
            f"{clause_block}\n\n"
            "Use these clauses to evaluate whether the query's request is approved or not.\n"
            "Your output must follow this JSON format:\n\n"
            "{\n"
            '  "query": "<user query>",\n'
            '  "decision": "<approved | rejected | insufficient info>",\n'
            '  "amount": "<amount if applicable>",\n'
            '  "justification": "<brief explanation of why this decision was made>",\n'
            '  "relevant_clauses": [\n'
            "    {\n"
            '      "file": "<doc name>",\n'
            '      "page": "<page number>",\n'
            '      "text": "<exact clause used>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "You MUST only base your decision on clauses retrieved above. If no clear answer is found, say \"insufficient info\". Do NOT hallucinate."
        )
        if edge_instruction:
            prompt += f"\n\n{edge_instruction}"
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

