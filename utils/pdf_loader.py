import fitz
from fastapi import UploadFile


class PDFLoader:
    """Utility class to extract text from uploaded PDF files."""

    async def extract_text(self, file: UploadFile) -> str:
        data = await file.read()
        with fitz.open(stream=data, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
