from app import LocalRAG


def test_rag_builds_index():
    rag = LocalRAG()
    count = rag.build()
    assert count > 0


def test_retrieval_returns_results():
    rag = LocalRAG()
    rag.build()
    results = rag.search("AI software developer productivity", top_k=3)
    assert len(results) > 0


def test_top_k_is_respected():
    rag = LocalRAG()
    rag.build()
    results = rag.search("software development", top_k=2)
    assert len(results) <= 2
