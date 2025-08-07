"""Vector store backed by FAISS for semantic search."""

from __future__ import annotations

import asyncio
from typing import Any, List, Dict


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
        self.metadatas: List[Dict] = []
        # Simple in-memory cache so repeated documents do not require
        # recomputing their embeddings on subsequent requests.
        self._cache: Dict[str, Any] = {}

    async def add_texts(self, texts: List[str], metadatas: List[Dict]) -> None:
        """Embed and index ``texts`` with their ``metadatas``."""

        if not texts:
            return

        missing = [t for t in texts if t not in self._cache]
        if missing:
            embeddings = await asyncio.to_thread(
                self.model.encode, missing, show_progress_bar=False
            )
            for text, emb in zip(missing, embeddings):
                self._cache[text] = self.np.array(emb, dtype="float32")

        vectors = self.np.array([self._cache[t] for t in texts]).astype("float32")
        if self.index is None:
            self.index = self.faiss.IndexFlatL2(vectors.shape[1])
        await asyncio.to_thread(self.index.add, vectors)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    async def similarity_search(
        self, query: str, k: int = 5, section: str | None = None
    ) -> List[Dict]:
        """Return the top ``k`` chunks most similar to ``query``.

        Optionally filter results to those matching a ``section`` tag.
        """

        if not self.index or not self.texts:
            return []
        embedding = await asyncio.to_thread(
            self.model.encode, [query], show_progress_bar=False
        )
        vector = self.np.array(embedding).astype("float32")
        search_k = max(k * 3, 10)
        distances, idxs = await asyncio.to_thread(self.index.search, vector, search_k)
        results: List[Dict] = []
        for dist, i in zip(distances[0], idxs[0]):
            if i < len(self.texts):
                meta = self.metadatas[i].copy()
                meta["text"] = self.texts[i]
                meta["score"] = float(1 / (1 + dist))
                if section and meta.get("section", "").lower() != section.lower():
                    continue
                results.append(meta)
                if len(results) >= k:
                    break
        return results
