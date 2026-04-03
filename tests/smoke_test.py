"""
DS Mentor Pro — End-to-end smoke test (11 subsystems).
Run with:  python tests/smoke_test.py
"""
import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agent import MentorAgent
from core.why_engine import WhyEngine
from core.question_generator import QuestionGenerator
from core.socratic_engine import SocraticEngine


class TestSmoke(unittest.TestCase):
    """Validates all 11 subsystems end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.agent = MentorAgent()

    def test_01_agent_initialization(self):
        self.assertIsNotNone(self.agent)

    def test_02_agent_process(self):
        res = self.agent.process_sync("How do I load a CSV?")
        self.assertIn("stage_num", res)
        self.assertTrue(1 <= res["stage_num"] <= 7)
        self.assertEqual(res["stage_name"], "Data Loading")
        self.assertIn("text", res)
        self.assertIn("code", res)

    def test_03_retrieval(self):
        context = self.agent.retriever.retrieve("load csv", active_stage=2)
        self.assertTrue(len(context) > 0)

    def test_04_stage_classification(self):
        stage, conf = self.agent.classifier.classify("train a random forest")
        self.assertEqual(stage, 6)
        self.assertTrue(0 < conf <= 1.0)

    def test_05_pipeline_tracking(self):
        self.agent.tracker.mark_completed(2)
        self.assertIn(2, self.agent.tracker.completed_stages)

    def test_06_session_memory(self):
        self.agent.memory.add_turn("user", "test q")
        self.agent.memory.add_turn("assistant", "test a")
        ctx = self.agent.memory.get_recent_context()
        self.assertIn("test q", ctx)

    def test_07_confidence_scoring(self):
        score = self.agent.scorer.score(
            retrieval_score=10.0, classifier_confidence=0.8,
            generated_code_valid=True, query_answer_overlap=0.3,
        )
        self.assertTrue(0 < score <= 1.0)

    def test_08_antipattern_detection(self):
        warnings = self.agent.detector.check_code(
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X = scaler.fit_transform(X)\n"
            "X.dropna()"
        )
        self.assertTrue(len(warnings) > 0)

    # -- New subsystems --

    def test_09_why_engine_enrichment(self):
        engine = WhyEngine()
        docs = [{"why_explanation": "Because variance", "when_to_use": "Always", "common_pitfall": "Leakage"}]
        result = engine.enrich({"text": "ans"}, docs, stage=4)
        self.assertEqual(result["why"], "Because variance")
        self.assertEqual(result["when_to_use"], "Always")

    def test_10_question_generation(self):
        res = self.agent.process_sync("How do I fill missing values in a numeric column?")
        suggestions = res.get("suggested_questions", [])
        self.assertIsInstance(suggestions, list)
        if suggestions:
            self.assertIn("question", suggestions[0])
            self.assertIn("type", suggestions[0])

    def test_11_socratic_mode(self):
        engine = SocraticEngine()
        engine.interaction_counter["smoke"] = engine.socratic_cooldown
        activated = engine.should_activate(
            "Why does scaling help models?", stage=4, skill_level=0.6, session_id="smoke"
        )
        self.assertTrue(activated)
        q = engine.generate_question("Why does scaling help?", stage=4)
        self.assertIn("socratic_question", q)
        self.assertTrue(len(q["socratic_question"]) > 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
