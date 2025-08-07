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
from fastapi.responses import JSONResponse
import uvicorn

from utils.auth import verify_token
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


async def _run_query_legacy(request: Request, token: str = Depends(get_current_user)):
    """Legacy implementation kept for reference."""

    content_type = request.headers.get("content-type", "")

    # ------------------------------------------------------------------
    # Parse input
    # ------------------------------------------------------------------
    questions: list[str] = []
    uploads: list[UploadFile] = []
    text_docs: list[tuple[str, str]] = []  # (content, name)

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # Questions can be under several possible keys
        q_list = payload.get("questions")
        if isinstance(q_list, list):
            questions.extend([q for q in q_list if isinstance(q, str)])
        for key in ("question", "query", "q"):
            val = payload.get(key)
            if isinstance(val, str):
                questions.append(val)

        doc_list = payload.get("documents", [])
        if isinstance(doc_list, list):
            for idx, doc in enumerate(doc_list):
                if isinstance(doc, str):
                    text_docs.append((doc, f"document_{idx}"))
    else:
        form = await request.form()

        # Extract question(s) regardless of the parameter name
        for key in ("question", "query", "q"):
            val = form.get(key)
            if isinstance(val, str):
                questions.append(val)
        questions.extend([q for q in form.getlist("questions") if isinstance(q, str)])

        # Collect files under "file" key (multiple allowed)
        for f in form.getlist("file"):
            if hasattr(f, "filename"):
                uploads.append(f)
        # Backwards compatibility: allow 'documents'
        for f in form.getlist("documents"):
            if hasattr(f, "filename"):
                uploads.append(f)

    if not questions:
        raise HTTPException(status_code=400, detail="No valid question provided")

    # ------------------------------------------------------------------
    # Validate document count and size
    # ------------------------------------------------------------------
    if len(uploads) + len(text_docs) > 10:
        raise HTTPException(status_code=400, detail="Too many documents")

    max_total_size = 200 * 1024 * 1024  # 200 MB
    total_size = 0

    # ------------------------------------------------------------------
    # Document processing and indexing
    # ------------------------------------------------------------------
    vector_store = VectorStore()

    # Process uploaded files
    for up in uploads:
        data = await up.read()
        total_size += len(data)
        if total_size > max_total_size:
            raise HTTPException(status_code=400, detail="Payload too large")
        chunks = await doc_loader.process_bytes(data, up.filename)
        texts = [c.text for c in chunks]
        metas = [c.metadata() for c in chunks]
        await vector_store.add_texts(texts, metas)

    # Process raw text documents from JSON payload
    for text, name in text_docs:
        data = text.encode("utf-8")
        total_size += len(data)
        if total_size > max_total_size:
            raise HTTPException(status_code=400, detail="Payload too large")
        chunks = await doc_loader.process_bytes(data, f"{name}.txt")
        texts = [c.text for c in chunks]
        metas = [c.metadata() for c in chunks]
        await vector_store.add_texts(texts, metas)

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
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
            parsed = json.loads(answer_raw)
            decision = parsed.get("decision", "")
            justification = parsed.get("justification", "")
        except Exception:
            decision = answer_raw
            justification = ""

        clauses = [
            {"file": r.get("file_name", ""), "text": r.get("text", "")}
            for r in retrieved
        ]
        return {
            "query": q,
            "decision": decision,
            "justification": justification,
            "relevant_clauses": clauses,
        }

    answers = await asyncio.gather(*(handle_question(q) for q in questions))

    return RAGResponse(status="success", answers=answers)


@app.post("/hackrx/run", response_model=RAGResponse)
async def run_query(request: Request, token: str = Depends(get_current_user)):
    """Endpoint that performs retrieval augmented generation over documents."""
    try:
        ct = request.headers.get("content-type", "")
        questions: list[str] = []
        uploads: list[UploadFile] = []
        text_docs: list[tuple[str, str]] = []
        if "application/json" in ct:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            q_list = payload.get("questions")
            if isinstance(q_list, list):
                questions.extend([q for q in q_list if isinstance(q, str)])
            for k in ("question", "query", "q"):
                v = payload.get(k)
                if isinstance(v, str):
                    questions.append(v)
            docs = payload.get("documents", [])
            if isinstance(docs, list):
                for i, d in enumerate(docs):
                    if isinstance(d, str):
                        text_docs.append((d, f"document_{i}"))
        else:
            form = await request.form()
            for k in ("question", "query", "q"):
                v = form.get(k)
                if isinstance(v, str):
                    questions.append(v)
            questions.extend([q for q in form.getlist("questions") if isinstance(q, str)])
            for f in form.getlist("file") + form.getlist("documents"):
                if hasattr(f, "filename"):
                    uploads.append(f)

        logger.info(
            "Payload received: questions=%s, uploads=%s, text_docs=%s",
            questions,
            [u.filename for u in uploads],
            [n for _, n in text_docs],
        )
        if not questions:
            raise HTTPException(status_code=400, detail="No valid question provided")
        if len(uploads) + len(text_docs) > 10:
            raise HTTPException(status_code=400, detail="Too many documents")

        max_size = 200 * 1024 * 1024
        total = 0
        store = VectorStore()

        datas = await asyncio.gather(*(u.read() for u in uploads))
        total += sum(len(d) for d in datas)
        if total > max_size:
            raise HTTPException(status_code=400, detail="Payload too large")
        tasks = [doc_loader.process_bytes(d, u.filename) for d, u in zip(datas, uploads)]
        for res in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(res, ValueError):
                raise HTTPException(status_code=415, detail=str(res))
            await store.add_texts([c.text for c in res], [c.metadata() for c in res])

        text_pairs = [(t.encode("utf-8"), f"{n}.txt") for t, n in text_docs]
        total += sum(len(d) for d, _ in text_pairs)
        if total > max_size:
            raise HTTPException(status_code=400, detail="Payload too large")
        tasks = [doc_loader.process_bytes(d, n) for d, n in text_pairs]
        for res in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(res, ValueError):
                raise HTTPException(status_code=415, detail=str(res))
            await store.add_texts([c.text for c in res], [c.metadata() for c in res])

        async def answer(q: str):
            logger.info("Processing query: %s", q)
            raw = await ollama_client.extract_entities(q)
            logger.info("Extracted entities: %s", raw)
            try:
                ent = json.loads(raw)
                kw = " ".join(str(v) for v in ent.values())
            except Exception:
                ent = {}
                kw = ""

            procedure = str(ent.get("procedure", ""))
            q_lower = q.lower()
            section_filter = None
            if any(w in q_lower for w in ["not cover", "not covered", "exclude", "exclusion", "rejected", "reject"]):
                section_filter = "exclusion"
            elif any(w in q_lower for w in ["cover", "covered", "claim", "approve", "approval", "inclusion", "include"]):
                section_filter = "inclusion"

            semantic_query = f"{q} {kw}".strip()
            retrieved = await store.similarity_search(semantic_query, k=5, section=section_filter)
            logger.info("Search results: %s", retrieved)

            procedure_found = False
            if procedure:
                proc_lower = procedure.lower()
                for r in retrieved:
                    if proc_lower in r.get("text", "").lower():
                        procedure_found = True
                        break

            inclusion_words = ["cover", "covered", "approve", "approved", "include"]
            exclusion_words = ["not covered", "exclude", "excluded", "reject", "rejected"]
            inclusion = any(any(w in r.get("text", "").lower() for w in inclusion_words) for r in retrieved)
            exclusion = any(any(w in r.get("text", "").lower() for w in exclusion_words) for r in retrieved)
            conflict = inclusion and exclusion

            clauses = [
                {
                    "file_name": r.get("file_name", ""),
                    "page_range": r.get("page_range", ""),
                    "text": r.get("text", ""),
                    "score": r.get("score", 0.0),
                }
                for r in retrieved
            ]

            if not clauses:
                return {
                    "query": q,
                    "decision": "insufficient info",
                    "amount": "",
                    "justification": "No relevant clauses were found in the provided documents.",
                    "relevant_clauses": [],
                    "confidence": "low",
                }

            if not procedure_found and procedure:
                just = (
                    f"The available clauses do not explicitly mention {procedure}. "
                    "Based on general hospitalization rules, the procedure may be covered if hospitalization is required, unless excluded elsewhere."
                )
                return {
                    "query": q,
                    "decision": "insufficient info",
                    "amount": "",
                    "justification": just,
                    "relevant_clauses": [
                        {
                            "file": c["file_name"],
                            "page": c["page_range"],
                            "text": c["text"],
                        }
                        for c in clauses
                    ],
                    "confidence": "low",
                }

            if conflict:
                just = "The policy has conflicting clauses. Further manual review is needed to determine approval status."
                return {
                    "query": q,
                    "decision": "insufficient info",
                    "amount": "",
                    "justification": just,
                    "relevant_clauses": [
                        {
                            "file": c["file_name"],
                            "page": c["page_range"],
                            "text": c["text"],
                        }
                        for c in clauses
                    ],
                    "confidence": "low",
                }

            edge_instruction = ""
            resp = await ollama_client.rag_answer(q, clauses, edge_instruction)
            logger.info("LLM response: %s", resp)
            try:
                parsed = json.loads(resp)
                dec = parsed.get("decision", "")
                amt = parsed.get("amount", "")
                just = parsed.get("justification", "")
            except Exception:
                dec = resp
                amt = ""
                just = ""

            best_score = max((c.get("score", 0.0) for c in clauses), default=0.0)
            if best_score > 0.8:
                conf = "high"
            elif best_score > 0.5:
                conf = "medium"
            else:
                conf = "low"

            result = {
                "query": q,
                "decision": dec,
                "amount": amt,
                "justification": just,
                "relevant_clauses": [
                    {
                        "file": c["file_name"],
                        "page": c["page_range"],
                        "text": c["text"],
                    }
                    for c in clauses
                ],
                "confidence": conf,
            }
            logger.info("Final answer: %s", result)
            return result

        ans = await asyncio.gather(*(answer(q) for q in questions))
        final = RAGResponse(status="success", answers=ans)
        logger.info("Final JSON returned: %s", final.model_dump())
        return final
    except HTTPException as exc:
        logger.exception("RAG endpoint error: %s", exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail},
        )
    except Exception as exc:
        logger.exception("Unexpected error in RAG endpoint", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
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
