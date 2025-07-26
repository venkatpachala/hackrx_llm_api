import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient:
    """Simple wrapper around the Google generative AI client."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = None
        logger.info("GeminiClient initialized")

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = genai.GenerativeModel("gemini-pro")

    async def answer_question(self, document_text: str, question: str) -> str:
        """Generate an answer for the given question based on the document."""
        prompt = (
            "You are an expert document analyzer. Answer the question based on the "
            "content provided. If the answer is not in the document, reply that it is not available.\n\n"
            f"DOCUMENT:\n{document_text}\n\nQUESTION: {question}\nANSWER:"
        )
        loop = asyncio.get_event_loop()
        try:
            self._ensure_model()
            response = await loop.run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            return response.text.strip() if response.text else ""
        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise

    async def test_connection(self) -> bool:
        """Check if the Gemini API is reachable."""
        try:
            resp = await self.answer_question("test", "reply with 'ok'")
            return bool(resp)
        except Exception:
            return False
