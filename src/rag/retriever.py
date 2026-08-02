"""
ChargeFlow AI V2 — RAG Similarity Retriever
=============================================
Retrieves relevant context chunks for a given user query.

Key Behaviors:
  - Embeds query string using fitted TextEmbedder
  - Searches VectorStore for top_k matches
  - Filters out matches with cosine similarity below configurable `similarity_threshold`
  - Returns structured source dicts with metadata and similarity scores
"""

from typing import Dict, List, Any, Optional
from src.rag.embedder import TextEmbedder
from src.rag.vector_store import VectorStore

DEFAULT_SIMILARITY_THRESHOLD = 0.40


class SimilarityRetriever:
    """
    Retrieves and filters knowledge chunks based on query vector similarity.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant chunks for query.

        Args:
            query: User's question string.
            top_k: Maximum chunks to retrieve.
            threshold: Override default similarity threshold if provided.

        Returns:
            List of source chunk dicts exceeding similarity_threshold.
            If no chunk meets the threshold, returns an empty list [].
        """
        if not query or not query.strip():
            return []

        if self.vector_store.is_empty():
            return []

        active_threshold = threshold if threshold is not None else self.similarity_threshold

        query_vec = self.embedder.transform([query])
        raw_results = self.vector_store.search(query_vec, top_k=top_k)

        filtered_sources = []
        for chunk, score in raw_results:
            if score >= active_threshold:
                filtered_sources.append({
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "section_title": chunk.get("section_title", "General"),
                    "text": chunk["text"],
                    "similarity_score": score,
                })

        return filtered_sources
