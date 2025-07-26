from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import List
import asyncio

from schemas import DocumentQueryRequest, DocumentQueryResponse
from utils.auth import verify_token
from utils.gemini_client import GeminiClient
from utils.pdf_loader import PDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HACKRX 6.0 - LLM Document Query API",
    description="An intelligent document understanding and retrieval system powered by Gemini Pro",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Initialize components
gemini_client = GeminiClient()
pdf_loader = PDFLoader()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token authentication"""
    if not verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HACKRX 6.0 - LLM Document Query API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test Gemini client
        gemini_status = await gemini_client.test_connection()
        return {
            "status": "healthy",
            "gemini_client": "connected" if gemini_status else "disconnected",
            "pdf_loader": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/hackrx/run", response_model=DocumentQueryResponse)
async def process_document_queries(
    request: DocumentQueryRequest,
    token: str = Depends(get_current_user)
):
    """
    Main endpoint for document understanding and query processing
    
    Takes documents and questions, returns intelligent answers using Gemini Pro
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Extract or process document text
        document_text = await pdf_loader.process_document(request.documents)
        
        if not document_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the document"
            )
        
        logger.info(f"Extracted {len(document_text)} characters from document")
        
        # Process questions concurrently for better performance
        tasks = [
            gemini_client.answer_question(document_text, question)
            for question in request.questions
        ]
        
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        processed_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                logger.error(f"Error processing question {i}: {str(answer)}")
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        logger.info(f"Successfully processed {len(processed_answers)} answers")
        
        return DocumentQueryResponse(answers=processed_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_document_queries: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )