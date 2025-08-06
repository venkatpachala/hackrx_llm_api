from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from io import BytesIO
from typing import List

from fastapi import UploadFile


@dataclass
class Chunk:
    """Represents a piece of text with tracing metadata."""

    chunk_id: int
    file_name: str
    page_range: str
    text: str

    def metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "page_range": self.page_range,
        }


class DocumentLoader:
    """Load multiple document types and chunk them for RAG."""

    def __init__(self, chunk_tokens: int = 1500, overlap_tokens: int = 200) -> None:
        """Create a new ``DocumentLoader``.

        Parameters
        ----------
        chunk_tokens:
            Approximate number of tokens (here treated as words) per chunk.  The
            default of ``1500`` is tuned for large documents (\u2265 1000 pages) so
            that each piece fed to the language model remains manageable.
        overlap_tokens:
            Number of words to overlap between consecutive chunks.  This helps
            maintain context continuity across chunk boundaries.
        """

        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    # --------------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        text = "".join(ch for ch in text if ch.isprintable())
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        size = self.chunk_tokens
        overlap = self.overlap_tokens
        start = 0
        chunks: List[str] = []
        while start < len(words):
            end = min(len(words), start + size)
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap
        return chunks

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    async def process_upload(self, upload: UploadFile) -> List[Chunk]:
        data = await upload.read()
        await upload.seek(0)
        return await self.process_bytes(data, upload.filename)

    async def process_base64(self, b64: str, name: str) -> List[Chunk]:
        try:
            data = base64.b64decode(b64)
        except Exception:
            return []
        return await self.process_bytes(data, name)

    async def process_bytes(self, data: bytes, file_name: str) -> List[Chunk]:
        lower = file_name.lower()
        if lower.endswith(".pdf") or data[:4] == b"%PDF":
            return await asyncio.to_thread(self._process_pdf, data, file_name)
        if lower.endswith(".docx"):
            return await asyncio.to_thread(self._process_docx, data, file_name)
        # Fallback to plain text
        return await asyncio.to_thread(
            self._process_text, data.decode("utf-8", errors="ignore"), file_name
        )

    # --------------------------------------------------------------
    # Handlers for specific formats
    # --------------------------------------------------------------
    def _process_pdf(self, data: bytes, file_name: str) -> List[Chunk]:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        chunks: List[Chunk] = []
        chunk_id = 0
        for page_idx, page in enumerate(doc):
            text = self._clean_text(page.get_text())
            for piece in self._chunk_text(text):
                chunks.append(Chunk(chunk_id, file_name, str(page_idx + 1), piece))
                chunk_id += 1
        return chunks

    def _process_docx(self, data: bytes, file_name: str) -> List[Chunk]:
        from docx import Document

        document = Document(BytesIO(data))
        text = "\n".join(p.text for p in document.paragraphs)
        cleaned = self._clean_text(text)
        chunks: List[Chunk] = []
        chunk_id = 0
        for piece in self._chunk_text(cleaned):
            chunks.append(Chunk(chunk_id, file_name, "1", piece))
            chunk_id += 1
        return chunks

    def _process_text(self, text: str, file_name: str) -> List[Chunk]:
        cleaned = self._clean_text(text)
        chunks: List[Chunk] = []
        chunk_id = 0
        for piece in self._chunk_text(cleaned):
            chunks.append(Chunk(chunk_id, file_name, "1", piece))
            chunk_id += 1
        return chunks
