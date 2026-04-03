"""Tests for the Socratic Engine."""
from core.socratic_engine import SocraticEngine


def test_should_activate_conceptual_intermediate():
    """Returns True for conceptual query from intermediate user."""
    engine = SocraticEngine()
    engine.interaction_counter["s1"] = engine.socratic_cooldown
    assert engine.should_activate("Why does normalization help models?", stage=4, skill_level=0.5, session_id="s1")


def test_should_not_activate_code_request():
    """Returns False for a code/how-to request."""
    engine = SocraticEngine()
    engine.interaction_counter["s2"] = engine.socratic_cooldown
    assert not engine.should_activate("Show me code for scaling features", stage=4, skill_level=0.5, session_id="s2")


def test_should_not_activate_beginner():
    """Returns False when skill < 0.3."""
    engine = SocraticEngine()
    engine.interaction_counter["s3"] = engine.socratic_cooldown
    assert not engine.should_activate("Why does normalization help?", stage=4, skill_level=0.1, session_id="s3")


def test_cooldown_respected():
    """Returns False if Socratic was used recently."""
    engine = SocraticEngine()
    engine.record_interaction("s4", was_socratic=True)
    engine.record_interaction("s4", was_socratic=False)
    assert not engine.should_activate("Why does this work?", stage=3, skill_level=0.6, session_id="s4")


def test_cooldown_resets_after_enough_interactions():
    """Returns True after cooldown period elapses."""
    engine = SocraticEngine()
    engine.record_interaction("s5", was_socratic=True)
    for _ in range(engine.socratic_cooldown):
        engine.record_interaction("s5", was_socratic=False)
    assert engine.should_activate("Explain the bias-variance tradeoff", stage=6, skill_level=0.6, session_id="s5")


def test_generate_question_returns_structure():
    """Has socratic_question, hint, full_answer_available."""
    engine = SocraticEngine()
    result = engine.generate_question("Why is scaling important?", stage=4)
    assert "socratic_question" in result
    assert "hint" in result
    assert result["full_answer_available"] is True
    assert len(result["socratic_question"]) > 10


def test_just_tell_me_bypasses():
    """'Just tell me' bypasses Socratic mode."""
    engine = SocraticEngine()
    engine.interaction_counter["s6"] = engine.socratic_cooldown
    assert not engine.should_activate("Just tell me why normalization helps", stage=4, skill_level=0.6, session_id="s6")


def test_is_conceptual_query():
    """Internal method correctly classifies queries."""
    engine = SocraticEngine()
    assert engine._is_conceptual_query("Why does feature scaling matter?")
    assert engine._is_conceptual_query("Explain the difference between bagging and boosting")
    assert not engine._is_conceptual_query("Show me how to load a CSV")
    assert not engine._is_conceptual_query("Write a function to compute F1 score")


def test_record_interaction_counter():
    """Counter increments and resets correctly."""
    engine = SocraticEngine()
    engine.record_interaction("t1", was_socratic=True)
    assert engine.interaction_counter["t1"] == 0

    engine.record_interaction("t1", was_socratic=False)
    assert engine.interaction_counter["t1"] == 1

    engine.record_interaction("t1", was_socratic=False)
    assert engine.interaction_counter["t1"] == 2
