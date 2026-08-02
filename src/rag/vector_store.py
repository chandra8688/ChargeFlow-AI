"""
ChargeFlow AI V2 — In-Memory NumPy Vector Store
=================================================
Simple, transparent, zero-dependency vector index.

Design:
  - Stores chunk metadata and an (N x D) normalized matrix of chunk vectors
  - Computes exact Cosine Similarity between query vector Q (1 x D) and all matrix rows:
        similarities = Q @ matrix.T
  - Fast, deterministic, sub-millisecond search for small to medium knowledge bases (< 500 chunks)
  - Pure Python + NumPy implementation
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class VectorStore:
    """
    In-memory vector store backing cosine similarity retrieval.
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: np.ndarray | None = None

    def build_index(self, chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
        """
        Populate vector store with chunk dicts and corresponding vector matrix.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but {len(vectors)} vectors passed."
            )
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Computes cosine similarity of query_vec against stored vector matrix.
        Returns top_k (chunk_dict, score) tuples sorted by score descending.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        # query_vec: (1, D) or (D,) -> reshape to (1, D)
        q = np.asarray(query_vec).reshape(1, -1)

        # Dot product of normalized vectors equals Cosine Similarity
        # Result shape: (1, N) -> flatten to (N,)
        sims = np.dot(q, self.vectors.T).ravel()

        # Sort indices descending by similarity score; tie-break deterministically by index
        indexed_scores = [(idx, float(sims[idx])) for idx in range(len(sims))]
        indexed_scores.sort(key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]))

        results = []
        for idx, score in indexed_scores[:top_k]:
            chunk_copy = dict(self.chunks[idx])
            results.append((chunk_copy, round(score, 4)))

        return results

    def is_empty(self) -> bool:
        return self.vectors is None or len(self.chunks) == 0

    def size(self) -> int:
        return len(self.chunks) if self.chunks else 0
