# test_adapter.py

from functools import lru_cache

from app import LocalRAG, generate_answer


@lru_cache(maxsize=1)
def get_rag() -> LocalRAG:
    """
    Build the RAG index once and reuse it for all tests.
    """
    rag = LocalRAG(
        chunk_size=900,
        overlap=150
    )
    rag.build()

    return rag


def retrieve(question: str, top_k: int = 5):
    """
    Test only the retrieval component of the RAG system.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    rag = get_rag()

    return rag.search(
        question,
        top_k=top_k
    )


def ask_chatbot(question: str, top_k: int = 5) -> str:
    """
    Run the complete RAG pipeline:
    retrieval + answer generation.
    """

    results = retrieve(
        question,
        top_k=top_k
    )

    answer = generate_answer(
        question,
        results
    )

    return answer