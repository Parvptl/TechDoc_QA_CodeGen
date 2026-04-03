"""Tests for the WHY Engine."""
from core.why_engine import WhyEngine


def test_enrich_returns_why_field():
    """Response has 'why' key after enrichment."""
    engine = WhyEngine()
    docs = [{"why_explanation": "Because it works", "when_to_use": "Always", "common_pitfall": "Don't skip it"}]
    result = engine.enrich({"text": "answer", "code": ""}, docs, stage=1)
    assert "why" in result
    assert result["why"] == "Because it works"


def test_enrich_returns_when_field():
    """Response has 'when_to_use' key."""
    engine = WhyEngine()
    docs = [{"why_explanation": "X", "when_to_use": "When data is clean", "common_pitfall": "Y"}]
    result = engine.enrich({"text": "a"}, docs, stage=2)
    assert result["when_to_use"] == "When data is clean"


def test_enrich_from_dataset():
    """When retrieved doc has why_explanation, it is used."""
    engine = WhyEngine(config={"mode": "dataset"})
    docs = [{"why_explanation": "Preserves variance", "when_to_use": "Use for normal data", "common_pitfall": "Leakage risk"}]
    result = engine.enrich({"text": "t"}, docs, stage=4)
    assert "Preserves variance" in result["why"]


def test_enrich_fallback_empty():
    """When doc has no why field, returns empty string (not crash)."""
    engine = WhyEngine()
    docs = [{"answer": "some answer"}]
    result = engine.enrich({"text": "t"}, docs, stage=3)
    assert result["why"] == ""
    assert result["when_to_use"] == ""
    assert result["common_pitfall"] == ""


def test_enrich_preserves_original():
    """Original text/code fields are unchanged."""
    engine = WhyEngine()
    docs = [{"why_explanation": "W", "when_to_use": "X", "common_pitfall": "Y"}]
    original = {"text": "original text", "code": "print(1)"}
    result = engine.enrich(original, docs, stage=1)
    assert result["text"] == "original text"
    assert result["code"] == "print(1)"


def test_enrich_empty_docs():
    """Empty retrieved docs returns empty enrichment without error."""
    engine = WhyEngine()
    result = engine.enrich({"text": "x"}, [], stage=5)
    assert result["why"] == ""


def test_parse_llm_output():
    """Static method parses structured LLM text."""
    text = "WHY: It reduces bias\nWHEN: Use for skewed data\nPITFALL: Don't forget scaling"
    result = WhyEngine._parse_llm_output(text)
    assert "reduces bias" in result["why"]
    assert "skewed data" in result["when_to_use"]
    assert "scaling" in result["common_pitfall"]
