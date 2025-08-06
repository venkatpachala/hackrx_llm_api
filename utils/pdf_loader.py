"""PDF loading utilities with streaming extraction and text chunking."""

from __future__ import annotations

import asyncio
from io import StringIO
from typing import List

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

    def __init__(self, chunk_size: int = 1000) -> None:
        self.chunk_size = chunk_size

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
                while len(buffer) >= self.chunk_size:
                    chunks.append(buffer[: self.chunk_size])
                    buffer = buffer[self.chunk_size :]
            if buffer:
                chunks.append(buffer)
            return chunks

        chunks = await asyncio.to_thread(_read_chunks)
        await file.seek(0)
        return chunks

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
        """Split plain text into ``chunk_size`` character segments."""

        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

