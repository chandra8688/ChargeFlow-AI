"""
ChargeFlow AI V2 — RAG Component & Service Unit Tests
======================================================
Tests for Phase 6 RAG pipeline:
  - DocumentLoader
  - TextChunker
  - TextEmbedder
  - VectorStore
  - SimilarityRetriever
  - PromptBuilder
  - RAGService facade (with injected MockLLMProvider)

100% deterministic, offline, zero paid API dependency.
"""

import unittest
import numpy as np
from src.rag.loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.embedder import TextEmbedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import SimilarityRetriever
from src.rag.prompt_builder import PromptBuilder
from src.rag.llm_provider import MockLLMProvider, GeminiLLMProvider
from src.rag.rag_service import RAGService


class TestRAGLoaderAndChunker(unittest.TestCase):

    def test_01_document_loader_reads_whitelisted_files(self):
        loader = DocumentLoader()
        docs = loader.load_documents()
        self.assertGreater(len(docs), 0, "Loader returned zero documents.")
        sources = [d["source"] for d in docs]
        self.assertTrue(any("README.md" in s for s in sources))
        self.assertTrue(any("problem_framing.md" in s for s in sources))

    def test_02_text_chunker_metadata_preservation(self):
        docs = [{"source": "test_doc.md", "content": "# Header\nThis is a sample test text.", "type": "markdown_doc"}]
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_documents(docs)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["source"], "test_doc.md")
        self.assertEqual(chunks[0]["chunk_id"], "test_doc_md_chunk_001")
        self.assertIn("Header", chunks[0]["section_title"])


class TestEmbedderAndVectorStore(unittest.TestCase):

    def test_03_embedder_fit_transform_normalized(self):
        embedder = TextEmbedder()
        texts = ["Demand forecasting uses lag features.", "Average utilization is 23%."]
        matrix = embedder.fit_transform(texts)
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(matrix.shape[1], 384, "Embedding dimension must be 384 for all-MiniLM-L6-v2")
        # Verify L2 norm is ~1.0 per row
        norms = np.linalg.norm(matrix, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4)

    def test_04_vector_store_cosine_similarity(self):
        embedder = TextEmbedder()
        texts = ["Demand forecasting model", "Charger utilization stats"]
        chunks = [
            {"chunk_id": "c1", "source": "s1", "text": texts[0]},
            {"chunk_id": "c2", "source": "s2", "text": texts[1]},
        ]
        matrix = embedder.fit_transform(texts)

        store = VectorStore()
        store.build_index(chunks, matrix)
        self.assertEqual(store.size(), 2)

        q_vec = embedder.transform(["forecast model"])
        results = store.search(q_vec, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0]["chunk_id"], "c1")
        self.assertGreater(results[0][1], results[1][1])


class TestRetrieverAndThreshold(unittest.TestCase):

    def setUp(self):
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedder = TextEmbedder()
        self.store = VectorStore()

        docs = self.loader.load_documents()
        chunks = self.chunker.chunk_documents(docs)
        matrix = self.embedder.fit_transform([c["text"] for c in chunks])
        self.store.build_index(chunks, matrix)

        self.retriever = SimilarityRetriever(
            embedder=self.embedder,
            vector_store=self.store,
            similarity_threshold=0.10,
        )

    def test_05_relevant_retrieval(self):
        sources = self.retriever.retrieve("features used in demand forecaster", top_k=3)
        self.assertGreater(len(sources), 0)
        self.assertIn("similarity_score", sources[0])
        self.assertGreaterEqual(sources[0]["similarity_score"], 0.10)

    def test_06_unsupported_question_rejection(self):
        # Query on completely unindexed topic should return zero sources at high threshold
        sources = self.retriever.retrieve("capital of France location city", top_k=3, threshold=0.40)
        self.assertEqual(len(sources), 0)

    def test_07_empty_query_returns_empty(self):
        sources = self.retriever.retrieve("", top_k=3)
        self.assertEqual(len(sources), 0)


class TestRAGServiceAndGrounding(unittest.TestCase):

    def setUp(self):
        mock_llm = MockLLMProvider()
        self.rag = RAGService(
            llm_provider=mock_llm,
            similarity_threshold=0.10,
        )
        self.rag.initialize()

    def test_08_supported_query_returns_grounded_answer(self):
        res = self.rag.query("What features does the forecasting model use?")
        self.assertTrue(res["grounded"])
        self.assertTrue(res["llm_invoked"])
        self.assertGreater(len(res["sources"]), 0)
        self.assertIn("16", res["answer"].lower() + " " + res["sources"][0]["text"].lower())

    def test_09_unsupported_query_triggers_refusal_no_llm_call(self):
        res = self.rag.query("What is quantum entanglement physics?", threshold=0.50)
        self.assertFalse(res["grounded"])
        self.assertFalse(res["llm_invoked"])
        self.assertEqual(len(res["sources"]), 0)
        self.assertIn("does not contain sufficient evidence", res["answer"])

    def test_10_no_real_provider_reports_unconfigured(self):
        unconfigured_rag = RAGService(
            llm_provider=GeminiLLMProvider(api_key=""),
            similarity_threshold=0.10,
        )
        unconfigured_rag.initialize()
        res = unconfigured_rag.query("What features does the forecasting model use?")
        self.assertTrue(res["grounded"])
        self.assertFalse(res["llm_invoked"])
        self.assertIn("unavailable because no real LLM provider", res["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
