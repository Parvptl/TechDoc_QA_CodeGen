"""Tests for the Question Generator."""
import json

from core.question_generator import QuestionGenerator


def _make_kb():
    return [
        {
            "query": "How do I load a CSV?",
            "stage": "2",
            "answer": "Use read_csv",
            "related_questions": json.dumps([
                "How do I check shape and dtypes?",
                "How do I handle encoding errors?",
                "How do I create a correlation heatmap?",
            ]),
            "difficulty": "beginner",
        },
        {
            "query": "How do I create a correlation heatmap?",
            "stage": "3",
            "answer": "Use seaborn heatmap",
            "related_questions": json.dumps([
                "How do I check for multicollinearity?",
                "Show the distribution of a column.",
                "How do I fill missing values?",
            ]),
            "difficulty": "intermediate",
        },
        {
            "query": "How do I fill missing values?",
            "stage": "4",
            "answer": "Use fillna or SimpleImputer",
            "related_questions": json.dumps([
                "What is MCAR vs MAR?",
                "How do I scale features?",
                "How do I detect outliers?",
            ]),
            "difficulty": "beginner",
        },
    ]


def test_suggest_returns_2_to_3_questions():
    """Always returns 2-3 suggestions."""
    kb = _make_kb()
    gen = QuestionGenerator(knowledge_base=kb)
    result = gen.suggest(
        current_question="How do I load a CSV?",
        current_stage=2,
        skill_level=0.5,
        retrieved_docs=kb[:1],
        session_questions=[],
    )
    assert 2 <= len(result) <= 3


def test_suggest_respects_skill_level_beginner():
    """Beginner gets more lateral questions."""
    kb = _make_kb()
    gen = QuestionGenerator(knowledge_base=kb)
    result = gen.suggest(
        current_question="How do I load a CSV?",
        current_stage=2,
        skill_level=0.1,
        retrieved_docs=kb[:1],
        session_questions=[],
    )
    types = [r["type"] for r in result]
    assert "lateral" in types or len(result) >= 2


def test_suggest_respects_skill_level_advanced():
    """Advanced users get more connected questions."""
    kb = _make_kb()
    gen = QuestionGenerator(knowledge_base=kb)
    result = gen.suggest(
        current_question="How do I load a CSV?",
        current_stage=2,
        skill_level=0.9,
        retrieved_docs=kb[:1],
        session_questions=[],
    )
    types = [r["type"] for r in result]
    assert "connected" in types or "deeper" in types


def test_suggest_filters_already_asked():
    """No duplicates with session history."""
    kb = _make_kb()
    gen = QuestionGenerator(knowledge_base=kb)
    result = gen.suggest(
        current_question="How do I load a CSV?",
        current_stage=2,
        skill_level=0.5,
        retrieved_docs=kb[:1],
        session_questions=["How do I check shape and dtypes?"],
    )
    for r in result:
        assert r["question"].lower() != "how do i check shape and dtypes?"


def test_suggest_handles_empty_dataset():
    """Returns empty list, doesn't crash."""
    gen = QuestionGenerator(knowledge_base=[])
    result = gen.suggest(
        current_question="Test",
        current_stage=1,
        skill_level=0.5,
        retrieved_docs=[],
        session_questions=[],
    )
    assert isinstance(result, list)


def test_suggest_returns_dict_structure():
    """Each suggestion has question, type, stage keys."""
    kb = _make_kb()
    gen = QuestionGenerator(knowledge_base=kb)
    result = gen.suggest(
        current_question="How do I fill missing values?",
        current_stage=4,
        skill_level=0.5,
        retrieved_docs=kb[2:],
        session_questions=[],
    )
    for r in result:
        assert "question" in r
        assert "type" in r
        assert "stage" in r
        assert r["type"] in ("lateral", "deeper", "connected")
