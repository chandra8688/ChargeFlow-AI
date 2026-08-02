"""
ChargeFlow AI V2 — RAG Retrieval Evaluation Script (Dense vs TF-IDF)
====================================================================
Evaluates actual Cosine Similarity retrieval scores across a benchmark set of:
  - Supported questions
  - Paraphrased supported questions
  - Borderline / subtle questions
  - Clearly unsupported / out-of-domain questions

Compares old TF-IDF sparse scores against new dense sentence embeddings ('all-MiniLM-L6-v2').
"""

from src.rag import DocumentLoader, TextChunker, TextEmbedder, VectorStore, SimilarityRetriever


OLD_TFIDF_SCORES = {
    "What features does the demand forecasting model use?": 0.1395,
    "What is the average charger utilization in India?": 0.1698,
    "How is the station recommender score calculated?": 0.1966,
    "What is the test R2 score of the demand forecaster?": 0.1229,
    "Which inputs influence the demand prediction model?": 0.1405,
    "How many public EV chargers are operational in India?": 0.2636,
    "How does ChargeFlow measure model error?": 0.1078,
    "What are the OCPP telemetry protocol details?": 0.1874,
    "How does the fleet manager user persona work?": 0.4826,
    "What is the capital city of France?": 0.2641,
    "Explain the principles of quantum entanglement.": 0.0000,
    "What is the battery chemistry of a Tesla Model Y?": 0.1190,
}

EVAL_BENCHMARK = [
    # 1. Supported questions
    ("SUPPORTED", "What features does the demand forecasting model use?"),
    ("SUPPORTED", "What is the average charger utilization in India?"),
    ("SUPPORTED", "How is the station recommender score calculated?"),
    ("SUPPORTED", "What is the test R2 score of the demand forecaster?"),

    # 2. Paraphrased supported questions
    ("PARAPHRASED", "Which inputs influence the demand prediction model?"),
    ("PARAPHRASED", "How many public EV chargers are operational in India?"),
    ("PARAPHRASED", "How does ChargeFlow measure model error?"),

    # 3. Borderline questions
    ("BORDERLINE", "What are the OCPP telemetry protocol details?"),
    ("BORDERLINE", "How does the fleet manager user persona work?"),

    # 4. Clearly unsupported / out-of-domain questions
    ("UNSUPPORTED", "What is the capital city of France?"),
    ("UNSUPPORTED", "Explain the principles of quantum entanglement."),
    ("UNSUPPORTED", "What is the battery chemistry of a Tesla Model Y?"),
]


def run_retrieval_evaluation():
    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = TextEmbedder()  # Now uses dense 'all-MiniLM-L6-v2'
    store = VectorStore()

    docs = loader.load_documents()
    chunks = chunker.chunk_documents(docs)
    matrix = embedder.fit_transform([c["text"] for c in chunks])
    store.build_index(chunks, matrix)

    retriever = SimilarityRetriever(embedder=embedder, vector_store=store, similarity_threshold=0.0)

    print("=" * 95)
    print("RAG DENSE EMBEDDING RETRIEVAL EVALUATION (all-MiniLM-L6-v2 vs TF-IDF)")
    print("=" * 95)
    print(f"Total Corpus Documents : {len(docs)}")
    print(f"Total Indexed Chunks   : {store.size()}")
    print(f"Embedding Dimension    : {embedder.dimension} (Dense Float32)")
    print("-" * 95)
    print(f"{'CATEGORY':<12} | {'OLD TF-IDF':<10} | {'DENSE SCORE':<11} | {'DELTA':<8} | QUESTION")
    print("-" * 95)

    results = []
    for cat, query in EVAL_BENCHMARK:
        matches = retriever.retrieve(query, top_k=1, threshold=0.0)
        old_score = OLD_TFIDF_SCORES.get(query, 0.0)
        if matches:
            top = matches[0]
            score = top["similarity_score"]
            src = top["source"]
        else:
            score = 0.0
            src = "NONE"

        delta = score - old_score
        print(f"{cat:<12} | {old_score:<10.4f} | {score:<11.4f} | {delta:<+8.4f} | {query}")
        results.append({"category": cat, "query": query, "old_score": old_score, "dense_score": score, "source": src})

    print("-" * 95)
    print("\nSUMMARY BY CATEGORY (DENSE DENSE EMBEDDINGS):")
    for cat in ["SUPPORTED", "PARAPHRASED", "BORDERLINE", "UNSUPPORTED"]:
        cat_dense = [r["dense_score"] for r in results if r["category"] == cat]
        cat_old = [r["old_score"] for r in results if r["category"] == cat]
        if cat_dense:
            avg_dense = sum(cat_dense) / len(cat_dense)
            min_dense = min(cat_dense)
            max_dense = max(cat_dense)
            avg_old = sum(cat_old) / len(cat_old)
            print(f"  {cat:<12} -> Old Avg: {avg_old:.4f} | New Dense Avg: {avg_dense:.4f} (Min: {min_dense:.4f}, Max: {max_dense:.4f})")

    print("=" * 95)
    return results


if __name__ == "__main__":
    run_retrieval_evaluation()
