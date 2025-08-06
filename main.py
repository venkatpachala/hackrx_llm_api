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
import base64
import hashlib
import json
from utils.document_loader import DocumentLoader
from utils.vector_store import VectorStore
from utils.ollama_client import OllamaClient
from schemas import RAGResponse
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
doc_loader = DocumentLoader()
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


@app.post("/hackrx/run", response_model=RAGResponse)
async def run_query(request: Request, token: str = Depends(get_current_user)):
    """Main RAG endpoint handling multiple documents and questions."""

    content_type = request.headers.get("content-type", "")
    questions: list[str] = []
    uploads: list[UploadFile] = []
    b64_docs: list[tuple[str, str]] = []  # (data, name)

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        q_list = payload.get("questions", [])
        if isinstance(q_list, list):
            questions.extend([q for q in q_list if isinstance(q, str)])
        doc_list = payload.get("documents", [])
        if isinstance(doc_list, list):
            for idx, doc in enumerate(doc_list):
                if isinstance(doc, dict):
                    data = doc.get("data") or doc.get("content")
                    name = doc.get("name", f"document_{idx}")
                else:
                    data = doc
                    name = f"document_{idx}"
                if isinstance(data, str):
                    b64_docs.append((data, name))
    else:
        form = await request.form()
        questions.extend([q for q in form.getlist("questions") if isinstance(q, str)])
        for file in form.getlist("documents"):
            if isinstance(file, UploadFile):
                uploads.append(file)

    if not questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    # Limit number and size of documents
    if len(uploads) + len(b64_docs) > 10:
        raise HTTPException(status_code=400, detail="Too many documents")

    total_size = 0
    for data, _ in b64_docs:
        total_size += len(data) * 3 // 4
    for up in uploads:
        await up.seek(0, 2)
        total_size += up.file.tell()
        await up.seek(0)
    if total_size > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Payload too large")

    # --------------------------------------------------------------
    # Document processing
    # --------------------------------------------------------------
    cache: dict[str, list] = {}

    async def process_bytes(data: bytes, name: str):
        key = hashlib.md5(data).hexdigest()
        if key in cache:
            return cache[key]
        chunks = await doc_loader.process_bytes(data, name)
        cache[key] = chunks
        return chunks

    tasks = []
    for up in uploads:
        data = await up.read()
        await up.seek(0)
        tasks.append(process_bytes(data, up.filename))
    for data, name in b64_docs:
        try:
            raw = base64.b64decode(data)
        except Exception:
            continue
        tasks.append(process_bytes(raw, name))

    chunk_lists = await asyncio.gather(*tasks)
    chunks = [c for lst in chunk_lists for c in lst]

    texts = [c.text for c in chunks]
    metas = [c.metadata() for c in chunks]

    vector_store = VectorStore()
    await vector_store.add_texts(texts, metas)

    # --------------------------------------------------------------
    # Question answering
    # --------------------------------------------------------------
    async def handle_question(q: str):
        entity_raw = await ollama_client.extract_entities(q)
        try:
            entities = json.loads(entity_raw)
            keywords = " ".join(str(v) for v in entities.values())
        except Exception:
            keywords = ""
        search_query = f"{q} {keywords}".strip()
        retrieved = await vector_store.similarity_search(search_query, k=5)
        context = "\n\n".join([r["text"] for r in retrieved])
        answer_raw = await ollama_client.rag_answer(q, context)
        try:
            answer = json.loads(answer_raw)
        except Exception:
            answer = {"decision": answer_raw, "amount": None, "justification": ""}
        return answer, retrieved

    results = await asyncio.gather(*(handle_question(q) for q in questions))
    answers = [res[0] for res in results]
    retrieved_all = [r for res in results for r in res[1]]

    retrieved_clauses = [
        {"file": r["file_name"], "page": r["page_range"], "clause": r["text"]}
        for r in retrieved_all
    ]

    return RAGResponse(
        status="success",
        answers=answers,
        retrieved_clauses=retrieved_clauses,
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
