from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os
import logging
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

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

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
if not API_TOKEN:
    raise ValueError("API_TOKEN environment variable is not set")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config={
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
)

# Pydantic models
class DocumentQueryRequest(BaseModel):
    documents: str = Field(..., min_length=1, description="Raw text content or base64 encoded PDF content")
    questions: List[str] = Field(..., min_items=1, max_items=50, description="List of questions to ask about the document")

class DocumentQueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the input questions")

# Authentication function
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Helper function to answer questions
async def answer_question(document_text: str, question: str) -> str:
    try:
        # Truncate document if too long
        max_doc_length = 30000
        if len(document_text) > max_doc_length:
            document_text = document_text[:max_doc_length] + "... [Document truncated]"
        
        prompt = f"""You are an expert document analyzer. Your task is to carefully read the provided document and answer the specific question based ONLY on the information contained within the document.

DOCUMENT CONTENT:
{document_text}

QUESTION: {question}

INSTRUCTIONS:
1. Read the document carefully and understand its content
2. Answer the question based ONLY on information found in the document
3. If the answer is not in the document, clearly state "The information is not available in the provided document"
4. Be specific and cite relevant details from the document when possible
5. Keep your answer concise but comprehensive

ANSWER:"""
        
        # Run the generation in a thread pool to avoid blocking
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )
        
        if not response.text:
            return "I couldn't generate an answer for this question. Please try rephrasing it."
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error answering question '{question}': {str(e)}")
        return f"Error processing question: {str(e)}"

# Routes
@app.get("/")
async def root():
    return {
        "message": "HACKRX 6.0 - LLM Document Query API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    try:
        # Test Gemini connection
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content("Hello, can you respond with 'Connected'?")
        )
        
        gemini_status = "connected" if response.text and "connected" in response.text.lower() else "disconnected"
        
        return {
            "status": "healthy",
            "gemini_client": gemini_status,
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
    token: str = Depends(verify_token)
):
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        document_text = request.documents
        
        if not document_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the document"
            )
        
        logger.info(f"Processing document with {len(document_text)} characters")
        
        # Process questions concurrently for better performance
        tasks = [
            answer_question(document_text, question)
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
    import uvicorn
    uvicorn.run(
        "main_working:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )