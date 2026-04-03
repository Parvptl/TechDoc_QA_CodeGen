from core.retriever import HybridRetriever


def test_retriever_returns_ranked_docs():
    docs = [
        {"query": "load csv file", "answer": "Use read_csv", "stage": 2, "code": ""},
        {"query": "train random forest", "answer": "Use sklearn", "stage": 6, "code": ""},
    ]
    r = HybridRetriever(stage_boost=2.0)
    r.add_documents(docs)
    out = r.retrieve("how to load csv", active_stage=2, top_k=2)
    assert out
    assert out[0]["stage"] == 2
    assert "retrieval_score" in out[0]


def test_retriever_stats_include_stage_distribution():
    r = HybridRetriever()
    r.add_documents([{"query": "q", "answer": "a", "stage": 1, "code": ""}])
    stats = r.get_stats()
    assert stats["total_documents"] == 1
