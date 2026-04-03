"""
MentorAgent -- ReAct-style orchestrator for DS Mentor Pro.
Integrates: retriever, classifier, pipeline tracker, code engine,
antipattern detector, confidence scorer, session memory, skill assessor,
WHY engine, question generator, Socratic engine, and dataset profiler.
"""
import os
import csv
import re

from storage.session_store import SessionStore

from .retriever import HybridRetriever
from .code_engine import CodeEngine
from .generator import Generator
from .stage_classifier import StageClassifier
from .pipeline_tracker import PipelineTracker
from .confidence_scorer import ConfidenceScorer
from .antipattern_detector import AntiPatternDetector
from .memory import SessionMemory
from .skill_assessor import SkillAssessor
from .why_engine import WhyEngine
from .question_generator import QuestionGenerator
from .socratic_engine import SocraticEngine
from .dataset_profiler import DatasetProfiler


class MentorAgent:
    """Orchestrates all DS Mentor subsystems in a sequential ReAct-style loop."""

    def __init__(self, data_docs=None):
        if data_docs is None:
            data_docs = self._load_default_docs()

        self.retriever = HybridRetriever()
        if data_docs:
            self.retriever.add_documents(data_docs)

        self.code_engine = CodeEngine()
        self.generator = Generator()
        self.classifier = StageClassifier()
        self.tracker = PipelineTracker()
        self.scorer = ConfidenceScorer()
        self.detector = AntiPatternDetector()

        self.memory = SessionMemory()
        self.assessor = SkillAssessor(self.memory)
        self.session_store = SessionStore()

        self.why_engine = WhyEngine()
        self.question_generator = QuestionGenerator(knowledge_base=data_docs or [])
        self.socratic_engine = SocraticEngine()

        self.dataset_profiler = DatasetProfiler()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_default_docs():
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(repo_root, "data", "dataset.csv")
        if not os.path.exists(dataset_path):
            return []
        with open(dataset_path, "r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    @staticmethod
    def _token_set(text: str):
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def _query_answer_overlap(self, query: str, answer: str) -> float:
        q_tokens = self._token_set(query)
        a_tokens = self._token_set(answer)
        if not q_tokens or not a_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / float(len(q_tokens))

    def _session_questions(self) -> list:
        return [m["content"] for m in self.memory.history if m.get("role") == "user"]

    # ------------------------------------------------------------------
    # dataset upload
    # ------------------------------------------------------------------

    def upload_dataset(self, file_bytes: bytes, filename: str) -> dict:
        """Profile an uploaded dataset and store it for context-aware responses."""
        profile = self.dataset_profiler.load_from_bytes(file_bytes, filename)
        self.memory.update_context("dataset_profile", profile)
        return profile

    # ------------------------------------------------------------------
    # main entry points
    # ------------------------------------------------------------------

    def process_sync(self, query: str, provided_code: str = "", session_id: str = "default") -> dict:
        """Synchronous process -- the full agent flow."""

        # Restore session
        existing = self.session_store.load(session_id)
        if existing:
            self.memory.history = existing.get("history", [])
            for k, v in existing.get("stage_skills", {}).items():
                try:
                    self.memory.profile.stage_skills[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            self.tracker.completed_stages = set(existing.get("completed_stages", []))

        # 1. Classify
        stage, class_conf = self.classifier.classify(query)
        stage_name = self.classifier.get_stage_name(stage)
        skill_level = self.memory.profile.stage_skills.get(stage, 0.5)

        # Socratic check -- before main flow
        if self.socratic_engine.should_activate(query, stage, skill_level, session_id):
            socratic = self.socratic_engine.generate_question(query, stage)
            self.socratic_engine.record_interaction(session_id, was_socratic=True)
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", socratic["socratic_question"])
            self._persist(session_id)
            return {
                "query": query,
                "stage_num": stage,
                "stage_name": stage_name,
                "pipeline_warnings": [],
                "antipattern_warnings": [],
                "confidence": 100.0,
                "text": socratic["socratic_question"],
                "code": "",
                "code_output": None,
                "difficulty_level": self.assessor.get_difficulty_string(stage),
                "mode": "socratic",
                "hint": socratic.get("hint", ""),
                "why": "",
                "when_to_use": "",
                "common_pitfall": "",
                "suggested_questions": [
                    {"question": "Just tell me the answer", "type": "direct", "stage": stage_name}
                ],
                "next_step": self.tracker.suggest_next_step(),
            }

        self.socratic_engine.record_interaction(session_id, was_socratic=False)

        # 2. Pipeline check
        warnings = self.tracker.check_prerequisites(stage)
        self.tracker.mark_completed(stage)

        # 3. Retrieve
        context = self.retriever.retrieve(query, active_stage=stage, top_k=3)
        retrieval_score = sum(d.get("retrieval_score", 0.0) for d in context) if context else 0.0

        # 4. Anti-pattern detect
        code_result = None
        antipattern_warnings = []
        is_code_valid = True

        if provided_code:
            antipattern_warnings = self.detector.check_code(provided_code)

        # 5. WHY enrichment
        why_data = self.why_engine.enrich(
            response={}, retrieved_docs=context, stage=stage,
        )

        # 6. Dataset context injection
        dataset_context = self.dataset_profiler.get_context_string()

        # 7. Generate response
        difficulty = self.assessor.get_difficulty_string(stage)
        gen_output = self.generator.generate(
            query=query,
            context=context,
            memory_context=self.memory.get_recent_context(),
            difficulty=difficulty,
            why_data=why_data,
            pipeline_warnings=warnings,
            dataset_context=dataset_context,
        )

        # 8. Execute code if present
        if gen_output.get("code"):
            code_result = self.code_engine.execute(gen_output["code"])
            is_code_valid = code_result["success"]

        # 9. Confidence
        overlap = self._query_answer_overlap(query, gen_output["text"])
        confidence = self.scorer.score(
            retrieval_score=retrieval_score,
            classifier_confidence=class_conf,
            generated_code_valid=is_code_valid,
            query_answer_overlap=overlap,
        )

        # 10. Skill update
        self.assessor.update_skill(stage, success=is_code_valid and not antipattern_warnings)

        # 11. Memory + persist
        self.memory.add_turn("user", query)
        self.memory.add_turn("assistant", gen_output["text"])
        self._persist(session_id)

        # 12. Follow-up suggestions
        suggestions = self.question_generator.suggest(
            current_question=query,
            current_stage=stage,
            skill_level=skill_level,
            retrieved_docs=context,
            session_questions=self._session_questions(),
        )

        return {
            "query": query,
            "stage_num": stage,
            "stage_name": stage_name,
            "pipeline_warnings": warnings,
            "antipattern_warnings": antipattern_warnings,
            "confidence": round(confidence * 100, 1),
            "text": gen_output["text"],
            "code": gen_output.get("code", ""),
            "code_output": code_result,
            "difficulty_level": difficulty,
            "mode": "direct",
            "hint": "",
            "why": why_data.get("why", ""),
            "when_to_use": why_data.get("when_to_use", ""),
            "common_pitfall": why_data.get("common_pitfall", ""),
            "suggested_questions": suggestions,
            "next_step": self.tracker.suggest_next_step(),
        }

    async def process(self, query: str, provided_code: str = "", session_id: str = "default") -> dict:
        """Async wrapper around process_sync."""
        return self.process_sync(query=query, provided_code=provided_code, session_id=session_id)

    def _persist(self, session_id: str):
        self.session_store.save(
            session_id,
            {
                "history": self.memory.history[-40:],
                "stage_skills": self.memory.profile.stage_skills,
                "completed_stages": list(self.tracker.completed_stages),
            },
        )
