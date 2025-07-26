from pydantic import BaseModel, Field


class DocumentQueryResponse(BaseModel):
    status: str = Field(..., description="Status of the request")
    query: str = Field(..., description="User question")
    answer: str = Field(..., description="LLM generated answer")
    file_name: str = Field(..., description="Name of the uploaded file")
