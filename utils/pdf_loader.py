"""PDF loading utilities with streaming extraction and text chunking."""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import List, Tuple

from fastapi import UploadFile
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter


class PDFLoader:
    """Utility class to stream text from uploaded PDF files.

    The implementation avoids loading the entire file into memory at once by
    processing the PDF page-by-page. Text is chunked on the fly so extremely
    large documents can be handled without exhausting RAM.
    """

    def __init__(self, chunk_size: int = 1800) -> None:
        # default chunk size roughly 1500-2000 characters
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_text(text: str, chunk_size: int) -> Tuple[List[str], str]:
        """Split ``text`` into chunks trying to respect sentence boundaries.

        Returns a tuple of ``(chunks, remainder)`` where ``remainder`` is the
        leftover text that didn't reach ``chunk_size``.
        """

        chunks: List[str] = []
        while len(text) >= chunk_size:
            # Find the last sentence break before the chunk boundary
            split_at = text.rfind(". ", 0, chunk_size)
            if split_at == -1 or split_at < chunk_size * 0.5:
                # no good sentence break found, hard split
                split_at = chunk_size
            else:
                split_at += 1  # include the period
            chunks.append(text[:split_at].strip())
            text = text[split_at:].lstrip()
        return chunks, text

    async def extract_text_chunks(self, file: UploadFile) -> List[str]:
        """Extract text from ``file`` and return it as a list of chunks.

        Args:
            file: ``UploadFile`` pointing to a PDF document.

        Returns:
            A list of strings, each roughly ``chunk_size`` characters long.

        The heavy lifting is done inside ``asyncio.to_thread`` to prevent
        blocking the event loop during CPU intensive PDF parsing.
        """

        await file.seek(0)

        def _read_chunks() -> List[str]:
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            chunks: List[str] = []
            buffer = ""
            for page in PDFPage.get_pages(file.file):
                output = StringIO()
                device = TextConverter(rsrcmgr, output, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                interpreter.process_page(page)
                device.close()
                text = output.getvalue().replace("\n", " ")
                buffer += text
                new_chunks, buffer = PDFLoader._split_text(buffer, self.chunk_size)
                chunks.extend(new_chunks)
            if buffer:
                chunks.append(buffer.strip())
            return chunks

        chunks = await asyncio.to_thread(_read_chunks)
        await file.seek(0)
        return chunks

    def chunk_text(self, text: str, chunk_size: int | None = None) -> List[str]:
        """Split plain text into chunks respecting sentence boundaries."""

        size = chunk_size or self.chunk_size
        chunks, remainder = self._split_text(text, size)
        if remainder.strip():
            chunks.append(remainder.strip())
        return chunks

