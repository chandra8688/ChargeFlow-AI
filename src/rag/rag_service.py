"""
ChargeFlow AI V2 — RAG Service Facade
======================================
High-level service orchestrating document ingestion, indexing, retrieval,
grounding validation, prompt formatting, and LLM answer generation.

Design Rules:
  1. Decoupled from ML pipeline: does NOT alter or query DemandForecaster or FeatureService.
  2. Strict Grounding Logic:
       retrieval score < threshold -> DO NOT CALL LLM -> grounded = False -> return refusal response.
       retrieval score >= threshold -> construct prompt -> call LLM -> return sources used.
  3. No Mock Fallback in Real App:
       If no real LLM provider API key is present at runtime, retrieval executes,
       and service clearly reports that LLM generation is unconfigured.
  4. Configurable similarity threshold and top_k parameters.
"""

import time
from typing import Dict, List, Any, Optional
from src.rag.loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import TextEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import SimilarityRetriever, DEFAULT_SIMILARITY_THRESHOLD
from src.rag.prompt_builder import PromptBuilder
from src.rag.llm_provider import LLMProvider, GeminiLLMProvider, MockLLMProvider


class RAGService:
    """
    RAG Service Facade.
    """

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[TextEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[SimilarityRetriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_provider: Optional[LLMProvider] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or VectorStore()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_provider = llm_provider or GeminiLLMProvider()
        self.similarity_threshold = similarity_threshold

        self.retriever = retriever or SimilarityRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            similarity_threshold=self.similarity_threshold,
        )
        self.is_initialized = False

    def initialize(self) -> int:
        """
        Loads documents, chunks content, fits embedder, builds vector index.
        Returns number of indexed chunks.
        """
        documents = self.loader.load_documents()
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            self.is_initialized = False
            return 0

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.fit_transform(texts)
        self.vector_store.build_index(chunks, vectors)
        self.is_initialized = True

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes RAG query workflow under strict grounding rules.

        Returns dict:
            {
                "question": str,
                "answer": str,
                "grounded": bool,
                "confidence_score": float,
                "sources": List[Dict],
                "llm_invoked": bool,
                "latency_ms": float
            }
        """
        t0 = time.perf_counter()

        if not self.is_initialized:
            self.initialize()

        active_threshold = threshold if threshold is not None else self.similarity_threshold
        sources = self.retriever.retrieve(question, top_k=top_k, threshold=active_threshold)

        max_score = max([s["similarity_score"] for s in sources]) if sources else 0.0

        # GROUNDING RULE: If retrieval yields zero sources meeting threshold,
        # DO NOT call LLM. Return evidence-based refusal response.
        if not sources or max_score < active_threshold:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "question": question,
                "answer": (
                    "The available ChargeFlow AI knowledge base does not contain "
                    "sufficient evidence to answer this question."
                ),
                "grounded": False,
                "confidence_score": round(max_score, 4),
                "sources": [],
                "llm_invoked": False,
                "latency_ms": round(latency_ms, 2),
            }

        # Check if LLM provider is available
        if not self.llm_provider.is_available():
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "question": question,
                "answer": (
                    "Answer generation is currently unavailable because no real LLM provider "
                    "(e.g. GEMINI_API_KEY) is configured. However, relevant knowledge sources "
                    "were successfully retrieved below."
                ),
                "grounded": True,
                "confidence_score": round(max_score, 4),
                "sources": sources,
                "llm_invoked": False,
                "latency_ms": round(latency_ms, 2),
            }

        # Construct prompt and call LLM
        prompt = self.prompt_builder.build_prompt(question, sources)
        system_inst = self.prompt_builder.get_system_instruction()

        try:
            raw_answer = self.llm_provider.generate(prompt, system_inst)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "question": question,
                "answer": raw_answer,
                "grounded": True,
                "confidence_score": round(max_score, 4),
                "sources": sources,
                "llm_invoked": True,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "question": question,
                "answer": f"LLM generation failed: {exc}",
                "grounded": False,
                "confidence_score": round(max_score, 4),
                "sources": sources,
                "llm_invoked": False,
                "latency_ms": round(latency_ms, 2),
            }
