from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentQueryResponse(BaseModel):
    status: str = Field(..., description="Status of the request")
    query: List[str] = Field(default_factory=list, description="User questions")
    answer: List[str] = Field(default_factory=list, description="LLM generated answers")
    file_name: Optional[str] = Field(
        None, description="Name of the uploaded file"
    )
