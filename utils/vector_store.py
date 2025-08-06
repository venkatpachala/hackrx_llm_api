"""Vector store backed by FAISS for semantic search."""

from __future__ import annotations

import asyncio
from typing import Any, List

# Heavy dependencies are imported lazily in ``__init__`` so importing this
# module does not require them unless the vector store is actually used.


class VectorStore:
    """Stores text chunks and performs similarity search via embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        self.model = SentenceTransformer(model_name)
        self.faiss = faiss
        self.np = np
        self.index: Any | None = None
        self.texts: List[str] = []

    async def add_texts(self, texts: List[str]) -> None:
        """Embed and index ``texts``."""

        if not texts:
            return

        embeddings = await asyncio.to_thread(
            self.model.encode, texts, show_progress_bar=False
        )
        vectors = self.np.array(embeddings).astype("float32")
        if self.index is None:
            self.index = self.faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.texts.extend(texts)

    async def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """Return the top ``k`` chunks most similar to ``query``."""

        if not self.index or not self.texts:
            return []
        embedding = await asyncio.to_thread(
            self.model.encode, [query], show_progress_bar=False
        )
        vector = self.np.array(embedding).astype("float32")
        _, idxs = self.index.search(vector, k)
        return [self.texts[i] for i in idxs[0] if i < len(self.texts)]

