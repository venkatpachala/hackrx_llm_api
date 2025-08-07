from __future__ import annotations

"""Pydantic schemas for responses returned by the API."""

from typing import List

from pydantic import BaseModel, Field


class RelevantClause(BaseModel):
    """Snippet of source text that supports an answer."""

    file: str = Field(..., description="Source file name")
    page: str = Field("", description="Page number or range within the source")
    text: str = Field(..., description="Exact text snippet used for the answer")


class Answer(BaseModel):
    """Model representing the answer for a single query."""

    query: str = Field(..., description="Original question")
    decision: str = Field(..., description="Decision or high level answer")
    amount: str = Field("", description="Payout amount if applicable")
    justification: str = Field(..., description="Reasoning behind the decision")
    relevant_clauses: List[RelevantClause] = Field(
        default_factory=list,
        description="Document snippets that support the decision",
    )
    confidence: str = Field(
        "", description="Heuristic confidence level for the decision"
    )


class RAGResponse(BaseModel):
    """Overall response structure returned by the RAG endpoint."""

    status: str = Field(..., description="Status of the request")
    answers: List[Answer] = Field(
        default_factory=list, description="Answers for each submitted question"
    )

