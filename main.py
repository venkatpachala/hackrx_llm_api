import logging
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    UploadFile,
    File,
    Form,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.auth import verify_token
from utils.pdf_loader import PDFLoader
from utils.gemini_client import GeminiClient
from schemas import DocumentQueryResponse
from fastapi.openapi.utils import get_openapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variables from a .env file are available
load_dotenv()

app = FastAPI(
    title="HACKRX 6.0 - LLM Document Query API",
    description="An intelligent document understanding and retrieval system powered by Gemini Pro",
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
gemini_client = GeminiClient()


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
    status_ok = await gemini_client.test_connection()
    return {"status": "healthy" if status_ok else "unhealthy"}


@app.post("/hackrx/run", response_model=DocumentQueryResponse)
async def run_query(
    question: str = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(get_current_user),
):
    try:
        document_text = await pdf_loader.extract_text(file)
    except Exception as exc:
        logger.error("Failed to load document: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read document") from exc

    try:
        answer = await gemini_client.answer_question(document_text, question)
    except Exception as exc:
        logger.error("Gemini processing error: %s", exc)
        raise HTTPException(status_code=500, detail="LLM processing failed") from exc

    return DocumentQueryResponse(
        status="success",
        query=question,
        answer=answer,
        file_name=file.filename or "uploaded",
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
