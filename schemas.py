from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Answer(BaseModel):
    decision: str = Field(..., description="Approval decision")
    amount: Optional[str] = Field(None, description="Approved amount if applicable")
    justification: str = Field(..., description="Clause reference for the decision")


class RetrievedClause(BaseModel):
    file: str = Field(..., description="Source file name")
    page: str = Field(..., description="Page number or range")
    clause: str = Field(..., description="Text of the retrieved clause")


class RAGResponse(BaseModel):
    status: str = Field(..., description="Status of the request")
    answers: List[Answer] = Field(default_factory=list, description="Answers for each question")
    retrieved_clauses: List[RetrievedClause] = Field(
        default_factory=list, description="Clauses used for answering"
    )
