import json
from pathlib import Path

import pytest

CASES = json.loads(
    Path(__file__).with_name("test_cases.json").read_text(encoding="utf-8")
)
REQUIRED_CATEGORIES = {
    "supported",
    "ambiguous",
    "unsupported",
    "boundary",
    "retrieval_failure",
    "hallucination",
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_case_has_question_and_expected_oracle(case):
    assert isinstance(case["question"], str)
    assert isinstance(case["expected_keywords"], list)
    assert case["expected_behavior"].strip()
    assert case["why_it_matters"].strip()


def test_case_ids_are_unique():
    ids = [case["id"] for case in CASES]

    assert len(ids) == len(set(ids))


def test_all_required_categories_are_present():
    categories = {case["type"] for case in CASES}

    assert REQUIRED_CATEGORIES <= categories


def test_retriever_ranks_relevant_evidence_first():
    from app import Chunk, retrieve

    relevant = Chunk(
        "A survey of 81,000 Claude users studied the economics of AI.",
        "economics.pdf",
        1,
    )
    unrelated = Chunk("The report describes an unrelated topic.", "other.pdf", 1)

    results = retrieve(
        "How many people took part in the survey about the economics of AI?",
        [unrelated, relevant],
        top_k=2,
    )

    assert results[0][0] == relevant
    assert results[0][1] > 0


def test_retriever_returns_no_results_for_zero_overlap():
    from app import Chunk, retrieve

    chunks = [Chunk("A survey of Claude users studied AI economics.", "report.pdf", 1)]

    assert retrieve("Qweryx Zorvex glorp", chunks, top_k=5) == []


def test_answer_is_limited_to_one_sentence():
    from app import limit_to_one_sentence

    assert limit_to_one_sentence("First sentence [1]. Second sentence.") == "First sentence [1]."


def test_prompt_requires_one_sentence():
    from app import build_prompt

    prompt = build_prompt("Question?", [])
    assert "exactly one sentence" in prompt
    assert "never repeat the question" in prompt


def test_fallback_answers_count_question_from_evidence(monkeypatch):
    from app import Chunk, answer_question

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunk = Chunk("Our recent survey of 81,000 Claude users studied AI economics.", "report.pdf", 1)

    answer = answer_question("How many people took part in the survey?", [(chunk, 0.8)])

    assert answer == "The survey included 81,000 Claude users, according to the report [1]."


def test_echoed_model_question_uses_evidence_answer(monkeypatch):
    import sys
    from types import SimpleNamespace

    from app import Chunk, answer_question

    question = "How many people took part in the AI-economics survey?"
    chunk = Chunk("What 81,000 people told us about the economics of AI.", "report.pdf", 1)

    class MockOpenAI:
        def __init__(self, api_key):
            self.responses = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(output_text=question)
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=MockOpenAI))

    answer = answer_question(question, [(chunk, 0.8)])

    assert answer == "The survey included 81,000 people, according to the report [1]."


def test_adapter_rejects_blank_question():
    from test_adapter import retrieve

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve(" \t\n")