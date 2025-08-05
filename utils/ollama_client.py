import asyncio
import logging

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with a local Ollama model server."""

    def __init__(self, model: str = "gemma:2b") -> None:
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        logger.info("OllamaClient initialized with model %s", self.model)

    def ask(self, prompt: str) -> str:
        """Send a prompt to the Ollama API and return the response text."""
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as exc:
            logger.error("Ollama API error: %s", exc)
            raise

    async def answer_question(self, document_text: str, question: str) -> str:
        """Format the document and question into a prompt and query the model."""
        prompt = f"Document:\n{document_text}\n\nQuestion:\n{question}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.ask(prompt))

    async def test_connection(self) -> bool:
        """Basic check to see if the Ollama service is reachable."""
        try:
            resp = await self.answer_question("test", "reply with ok")
            return bool(resp)
        except Exception as exc:
            logger.error("Ollama connection test failed: %s", exc)
            return False
