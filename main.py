import asyncio
import logging
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    UploadFile,
    Request,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.auth import verify_token
from utils.pdf_loader import PDFLoader
from utils.ollama_client import OllamaClient
from schemas import DocumentQueryResponse
from fastapi.openapi.utils import get_openapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variables from a .env file are available
load_dotenv()

app = FastAPI(
    title="HACKRX 6.0 - LLM Document Query API",
    description="An intelligent document understanding and retrieval system powered by a local Ollama model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
pdf_loader = PDFLoader()
ollama_client = OllamaClient()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@app.get("/")
async def root():
    return {
        "message": "HACKRX 6.0 - LLM Document Query API",
        "status": "active",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    status_ok = await ollama_client.test_connection()
    return {"status": "healthy" if status_ok else "unhealthy"}


@app.post("/hackrx/run", response_model=DocumentQueryResponse)
async def run_query(request: Request, token: str = Depends(get_current_user)):
    raw_body = await request.body()
    logger.info("Raw request body: %s", raw_body)

    questions: list[str] = []
    text_blobs: list[str] = []
    file_name: str | None = None
    upload: UploadFile | None = None

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        q_single = payload.get("question")
        if isinstance(q_single, str):
            questions.append(q_single)
        q_list = payload.get("questions")
        if isinstance(q_list, list):
            questions.extend([q for q in q_list if isinstance(q, str)])
        docs = payload.get("documents")
        if isinstance(docs, str):
            text_blobs.append(docs)
    else:
        form = await request.form()
        q_single = form.get("question")
        if isinstance(q_single, str):
            questions.append(q_single)
        q_list = form.getlist("questions")
        for q in q_list:
            if isinstance(q, str):
                questions.append(q)
        docs = form.get("documents")
        if isinstance(docs, str):
            text_blobs.append(docs)
        possible_upload = form.get("file")
        if isinstance(possible_upload, UploadFile):
            upload = possible_upload
            file_name = upload.filename

    # ------------------------------------------------------------------
    # Prepare chunks from supplied text and/or uploaded PDF
    # ------------------------------------------------------------------
    chunks: list[str] = []

    for blob in text_blobs:
        chunks.extend(pdf_loader.chunk_text(blob))

    if upload is not None:
        try:
            pdf_chunks = await pdf_loader.extract_text_chunks(upload)
            chunks.extend(pdf_chunks)
            logger.info("Indexed %d chunks from %s", len(pdf_chunks), file_name)
        except Exception as exc:
            logger.error("Failed to load document: %s", exc)

    # ------------------------------------------------------------------
    # Question processing helpers
    # ------------------------------------------------------------------
    async def process_question(question: str) -> str:
        async def ask_chunk(chunk: str) -> str:
            try:
                return await asyncio.wait_for(
                    ollama_client.answer_question(chunk, question), timeout=15
                )
            except Exception as exc:
                logger.error("Chunk processing failed: %s", exc)
                return ""

        partial_answers = await asyncio.gather(*(ask_chunk(c) for c in chunks))
        partial_answers = [ans for ans in partial_answers if ans]

        if not partial_answers:
            return "No relevant information found."

        combined_context = "\n\n".join(partial_answers)
        try:
            return await asyncio.wait_for(
                ollama_client.answer_question(combined_context, question),
                timeout=15,
            )
        except Exception as exc:
            logger.error("Final summarization failed: %s", exc)
            # Return concatenated partial answers if final step fails
            return combined_context

    answers = await asyncio.gather(*(process_question(q) for q in questions))

    return DocumentQueryResponse(
        status="success",
        query=questions,
        answer=list(answers),
        file_name=file_name,
    )


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema.setdefault("components", {}).setdefault(
        "securitySchemes", {}).update(
        {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
            }
        }
    )
    for path in openapi_schema.get("paths", {}).values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    await ollama_client.aclose()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
