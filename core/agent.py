"""
MentorAgent -- ReAct-style orchestrator for DS Mentor Pro.
Integrates: retriever, classifier, pipeline tracker, code engine,
antipattern detector, confidence scorer, session memory, skill assessor,
WHY engine, question generator, Socratic engine, and dataset profiler.
"""
import os
import csv
import re
import time

from storage.session_store import SessionStore
from storage.learning_analytics import LearningAnalyticsStore
from storage.report_exporter import ReportExporter

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
from .agents.planner import PlannerAgent
from .agents.critic import CriticAgent
from .detection.misconception import MisconceptionDetector
from .retrieval.query_expand import QueryExpander
from .debug_assistant import DebugAssistant
from .project_mode import ProjectModeEngine
from .code_annotator import CodeAnnotator
from .checkpoint_assessor import CheckpointAssessor
from .quiz_engine import QuizEngine


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
        self.analytics = LearningAnalyticsStore()
        self.report_exporter = ReportExporter()

        self.why_engine = WhyEngine()
        self.question_generator = QuestionGenerator(knowledge_base=data_docs or [])
        self.socratic_engine = SocraticEngine()
        self.planner = PlannerAgent()
        self.critic = CriticAgent(retry_threshold=0.6)
        self.misconception_detector = MisconceptionDetector(min_hits=1)
        self.query_expander = QueryExpander()
        self.debug_assistant = DebugAssistant()

        self.dataset_profiler = DatasetProfiler()
        self.project_mode = ProjectModeEngine()
        self.code_annotator = CodeAnnotator()
        self.checkpoint_assessor = CheckpointAssessor(min_required_score=0.55)
        self.quiz_engine = QuizEngine()

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
        self.memory.update_context(
            "project_plan",
            self.project_mode.generate_plan(
                profile,
                self.memory.profile.stage_skills,
                completed_stages=self.tracker.completed_stages,
            ),
        )
        return profile

    def get_project_plan(self) -> dict:
        profile = self.memory.active_context.get("dataset_profile")
        if not profile:
            return {
                "mode": "project",
                "title": "Project mode unavailable",
                "brief": "Upload a dataset to generate a guided project plan.",
                "dataset_summary": {},
                "checkpoints": [],
            }
        return self.project_mode.generate_plan(
            profile,
            self.memory.profile.stage_skills,
            completed_stages=self.tracker.completed_stages,
        )

    def get_progress(self, session_id: str = "default") -> dict:
        existing = self.session_store.load(session_id)
        stage_skills = existing.get("stage_skills", {}) if existing else {}
        completed = existing.get("completed_stages", []) if existing else []
        analytics = self.analytics.session_summary(session_id)
        mastery = {}
        for i in range(1, 8):
            raw = stage_skills.get(str(i), stage_skills.get(i, self.memory.profile.stage_skills.get(i, 0.5)))
            mastery[i] = round(float(raw), 2)
        return {
            "session_id": session_id,
            "stage_mastery": mastery,
            "completed_stages": sorted(int(s) for s in completed),
            "questions_asked": analytics["query_count"],
            "misconceptions": analytics["misconceptions_total"],
            "antipattern_warnings": analytics["antipattern_total"],
            "avg_confidence": analytics["avg_confidence"],
            "avg_response_ms": analytics["avg_response_ms"],
            "quiz_count": analytics["quiz_count"],
            "avg_quiz_score": analytics["avg_quiz_score"],
            "checkpoint_passed": analytics["checkpoint_passed"],
            "checkpoint_total": analytics["checkpoint_total"],
            "stage_distribution": analytics["stage_query_distribution"],
        }

    def get_learning_dashboard(self, session_id: str = "default") -> dict:
        summary = self.get_progress(session_id)
        trends = self.analytics.session_trends(session_id)
        return {"summary": summary, "trends": trends}

    def export_learning_report(self, session_id: str = "default", fmt: str = "markdown") -> dict:
        report_data = self.get_learning_dashboard(session_id)
        export = self.report_exporter.export(session_id=session_id, report_data=report_data, fmt=fmt)
        return {
            "session_id": session_id,
            "format": export["format"],
            "path": export["path"],
            "content": export["content"],
        }

    def generate_quiz(self, session_id: str = "default", stage: int = None, difficulty: str = None) -> dict:
        progress = self.get_progress(session_id)
        use_stage = int(stage) if stage else self._next_focus_stage(progress.get("stage_mastery", {}))
        stage_name = self.classifier.get_stage_name(use_stage)
        use_difficulty = difficulty or self.assessor.get_difficulty_string(use_stage)
        questions = self.quiz_engine.generate(stage_name)
        return {
            "session_id": session_id,
            "stage": use_stage,
            "difficulty": use_difficulty,
            "questions": questions,
        }

    def grade_quiz(self, session_id: str, answers: list, stage: int = None) -> dict:
        quiz = self.generate_quiz(session_id=session_id, stage=stage)
        result = self.quiz_engine.grade(quiz["questions"], answers or [])
        self.assessor.update_skill(stage=quiz["stage"], success=result.score >= 0.67)
        self._persist(session_id)
        self.analytics.log_quiz_event(
            session_id=session_id,
            stage_num=int(quiz["stage"]),
            score=float(result.score),
            correct=int(result.correct),
            total=int(result.total),
        )
        return {
            "session_id": session_id,
            "stage": quiz["stage"],
            "score": result.score,
            "correct": result.correct,
            "total": result.total,
            "details": result.details,
            "mastery_updated": True,
        }

    def submit_project_checkpoint(self, session_id: str, stage_num: int, evidence: str) -> dict:
        existing = self.session_store.load(session_id) or {}
        completed = set(existing.get("completed_stages", []))
        next_required = self.project_mode.next_required_stage(completed)
        if int(stage_num) != next_required:
            self.analytics.log_checkpoint_event(session_id, int(stage_num), accepted=False, score=0.0)
            return {
                "accepted": False,
                "message": f"Submit Stage {next_required} checkpoint next.",
                "next_required_stage": next_required,
            }

        assessment = self.checkpoint_assessor.assess(stage_num, evidence)
        if not assessment.passed:
            self.analytics.log_checkpoint_event(
                session_id, int(stage_num), accepted=False, score=float(assessment.score)
            )
            return {
                "accepted": False,
                "message": assessment.feedback,
                "next_required_stage": next_required,
                "assessment": {
                    "score": assessment.score,
                    "min_required_score": assessment.min_required_score,
                    "matched_keywords": assessment.matched_keywords,
                },
            }

        completed.add(int(stage_num))
        existing["completed_stages"] = sorted(int(x) for x in completed)
        if "history" not in existing:
            existing["history"] = []
        if "stage_skills" not in existing:
            existing["stage_skills"] = self.memory.profile.stage_skills
        if "active_context" not in existing:
            existing["active_context"] = self.memory.active_context
        self.session_store.save(session_id, existing)

        self.tracker.completed_stages = set(existing["completed_stages"])
        self.analytics.log_checkpoint_event(
            session_id, int(stage_num), accepted=True, score=float(assessment.score)
        )
        updated_plan = self.get_project_plan()
        return {
            "accepted": True,
            "message": f"Stage {int(stage_num)} checkpoint accepted.",
            "next_required_stage": self.project_mode.next_required_stage(existing["completed_stages"]),
            "assessment": {
                "score": assessment.score,
                "min_required_score": assessment.min_required_score,
                "matched_keywords": assessment.matched_keywords,
            },
            "project_plan": updated_plan,
        }

    def _log_event(self, session_id: str, result: dict, response_time_ms: int):
        self.analytics.log_event(
            session_id=session_id,
            stage_num=int(result.get("stage_num", 1)),
            confidence=float(result.get("confidence", 0.0)),
            confidence_label=str(result.get("confidence_label", "")),
            misconceptions_count=len(result.get("misconception_alerts", [])),
            antipattern_count=len(result.get("antipattern_warnings", [])),
            response_time_ms=int(response_time_ms),
            mode=str(result.get("mode", "direct")),
        )

    @staticmethod
    def _next_focus_stage(stage_mastery: dict) -> int:
        if not stage_mastery:
            return 1
        parsed = []
        for k, v in stage_mastery.items():
            try:
                parsed.append((int(k), float(v)))
            except (TypeError, ValueError):
                continue
        if not parsed:
            return 1
        parsed.sort(key=lambda x: x[1])
        return int(parsed[0][0])

    # ------------------------------------------------------------------
    # main entry points
    # ------------------------------------------------------------------

    def process_sync(self, query: str, provided_code: str = "", session_id: str = "default") -> dict:
        """Synchronous process -- the full agent flow."""
        t0 = time.perf_counter()

        # Restore session
        existing = self.session_store.load(session_id)
        if existing:
            self.memory.history = existing.get("history", [])
            self.memory.active_context = existing.get("active_context", {})
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
        difficulty = self.assessor.get_difficulty_string(stage)

        # Debug-first route when user submits traceback/error messages.
        if self.debug_assistant.looks_like_traceback(query):
            dbg = self.debug_assistant.suggest(query)
            text = (
                f"**Debug diagnosis**\n"
                f"- Error type: {dbg.error_type}\n"
                f"- Category: {dbg.category}\n"
                f"- Line: {dbg.line_number if dbg.line_number > 0 else 'unknown'}\n"
                f"- Likely cause: {dbg.likely_cause}\n\n"
                f"**Recommended fix**\n{dbg.fix}"
            )
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", text)
            self._persist(session_id)
            result = {
                "query": query,
                "stage_num": stage,
                "stage_name": stage_name,
                "pipeline_warnings": [],
                "antipattern_warnings": [],
                "misconception_alerts": [],
                "confidence": 72.0,
                "confidence_label": "Moderate confidence - please verify",
                "text": text,
                "code": "",
                "code_output": None,
                "difficulty_level": difficulty,
                "mode": "debug",
                "hint": "",
                "why": "",
                "when_to_use": "",
                "common_pitfall": "",
                "suggested_questions": [],
                "next_step": self.tracker.suggest_next_step(),
                "plan_subtasks": [],
                "pedagogy_mode": "direct",
                "code_explained": "",
                "critic": {
                    "factual_accuracy": 0.75,
                    "pedagogical_fit": 0.75,
                    "completeness": 0.65,
                    "feedback": "Traceback parsed with known error patterns.",
                    "retried": False,
                },
            }
            self._log_event(session_id, result, int((time.perf_counter() - t0) * 1000))
            return result

        # Planner decomposition and misconception detection
        plan = self.planner.plan(
            query=query,
            stage=stage,
            skill_level=skill_level,
            history=self._session_questions(),
        )
        misconceptions = self.misconception_detector.detect(query=query, provided_code=provided_code, stage=stage)
        misconception_corrections = [m["correction"] for m in misconceptions]

        # Project-mode gating: if dataset-backed project mode is active, force sequential stage progression.
        if self.memory.active_context.get("dataset_profile"):
            gate = self.project_mode.gate_stage_jump(stage, self.tracker.completed_stages)
            if gate.get("blocked"):
                next_stage = int(gate["next_required_stage"])
                checkpoint_prompt = self.project_mode._checkpoint_text(next_stage, self.dataset_profiler.profile.get("target_guess") if self.dataset_profiler.profile else "target")
                text = (
                    f"**Project checkpoint required**\n{gate['message']}\n\n"
                    f"**Next checkpoint**\n{checkpoint_prompt}\n\n"
                    f"Ask a question focused on Stage {next_stage} to unlock the next stage."
                )
                self.memory.add_turn("user", query)
                self.memory.add_turn("assistant", text)
                self._persist(session_id)
                result = {
                    "query": query,
                    "stage_num": stage,
                    "stage_name": stage_name,
                    "pipeline_warnings": [gate["message"]],
                    "antipattern_warnings": [],
                    "misconception_alerts": misconceptions,
                    "confidence": 85.0,
                    "confidence_label": "High confidence",
                    "text": text,
                    "code": "",
                    "code_explained": "",
                    "code_output": None,
                    "difficulty_level": difficulty,
                    "mode": "project_gate",
                    "hint": "",
                    "why": "",
                    "when_to_use": "",
                    "common_pitfall": "",
                    "suggested_questions": [
                        {
                            "question": f"Help me with Stage {next_stage}",
                            "type": "checkpoint",
                            "stage": self.classifier.get_stage_name(next_stage),
                        }
                    ],
                    "next_step": self.tracker.suggest_next_step(),
                    "plan_subtasks": [s.description for s in plan.subtasks],
                    "pedagogy_mode": plan.pedagogy_mode,
                    "critic": {
                        "factual_accuracy": 1.0,
                        "pedagogical_fit": 0.9,
                        "completeness": 0.9,
                        "feedback": "Project mode gate enforced to preserve learning sequence.",
                        "retried": False,
                    },
                }
                self._log_event(session_id, result, int((time.perf_counter() - t0) * 1000))
                return result

        # Socratic check -- before main flow
        if self.socratic_engine.should_activate(query, stage, skill_level, session_id):
            socratic = self.socratic_engine.generate_question(query, stage)
            self.socratic_engine.record_interaction(session_id, was_socratic=True)
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", socratic["socratic_question"])
            self._persist(session_id)
            result = {
                "query": query,
                "stage_num": stage,
                "stage_name": stage_name,
                "pipeline_warnings": [],
                "antipattern_warnings": [],
                "confidence": 100.0,
                "text": socratic["socratic_question"],
                "code": "",
                "code_output": None,
                "difficulty_level": difficulty,
                "mode": "socratic",
                "hint": socratic.get("hint", ""),
                "why": "",
                "when_to_use": "",
                "common_pitfall": "",
                "misconception_alerts": misconceptions,
                "plan_subtasks": [s.description for s in plan.subtasks],
                "pedagogy_mode": plan.pedagogy_mode,
                "confidence_label": "High confidence",
                "code_explained": "",
                "critic": {
                    "factual_accuracy": 1.0,
                    "pedagogical_fit": 1.0,
                    "completeness": 1.0,
                    "feedback": "Socratic path selected.",
                    "retried": False,
                },
                "suggested_questions": [
                    {"question": "Just tell me the answer", "type": "direct", "stage": stage_name}
                ],
                "next_step": self.tracker.suggest_next_step(),
            }
            self._log_event(session_id, result, int((time.perf_counter() - t0) * 1000))
            return result

        self.socratic_engine.record_interaction(session_id, was_socratic=False)

        # 2. Pipeline check
        warnings = self.tracker.check_prerequisites(stage)
        self.tracker.mark_completed(stage)

        # 3. Retrieve
        expanded_query = self.query_expander.expand(query, extra_terms=plan.key_terms)
        context = self.retriever.retrieve(
            expanded_query,
            active_stage=stage,
            top_k=3,
            skill_level=skill_level,
        )
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
        gen_output = self.generator.generate(
            query=query,
            context=context,
            memory_context=self.memory.get_recent_context(),
            difficulty=difficulty,
            why_data=why_data,
            pipeline_warnings=warnings,
            dataset_context=dataset_context,
            pedagogy_mode=plan.pedagogy_mode,
            planner_constraints=plan.constraints,
            misconception_corrections=misconception_corrections,
        )

        # 7b. Critic checks with single retry
        required_subtasks = [sub.kind for sub in plan.subtasks]
        critic_score = self.critic.evaluate(
            response_text=gen_output["text"],
            retrieved_context=context,
            query=query,
            difficulty=difficulty,
            required_subtasks=required_subtasks,
        )
        retried = False
        if self.critic.should_retry(critic_score):
            retried = True
            gen_output = self.generator.generate(
                query=query,
                context=context,
                memory_context=self.memory.get_recent_context(),
                difficulty=difficulty,
                why_data=why_data,
                pipeline_warnings=warnings,
                dataset_context=dataset_context,
                pedagogy_mode=plan.pedagogy_mode,
                planner_constraints=plan.constraints,
                misconception_corrections=misconception_corrections,
                retry_feedback=critic_score.feedback,
            )
            critic_score = self.critic.evaluate(
                response_text=gen_output["text"],
                retrieved_context=context,
                query=query,
                difficulty=difficulty,
                required_subtasks=required_subtasks,
            )

        # 8. Execute code if present
        if gen_output.get("code"):
            code_result = self.code_engine.execute(gen_output["code"])
            is_code_valid = code_result["success"]
        explained_code = self.code_annotator.annotate(gen_output.get("code", ""))

        # 9. Confidence
        overlap = self._query_answer_overlap(query, gen_output["text"])
        confidence = self.scorer.score(
            retrieval_score=retrieval_score,
            classifier_confidence=class_conf,
            generated_code_valid=is_code_valid,
            query_answer_overlap=overlap,
            critic_min_axis=critic_score.min_axis,
        )
        confidence_parts = self.scorer.score_components(
            retrieval_score=retrieval_score,
            classifier_confidence=class_conf,
            generated_code_valid=is_code_valid,
            query_answer_overlap=overlap,
            critic_min_axis=critic_score.min_axis,
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

        result = {
            "query": query,
            "stage_num": stage,
            "stage_name": stage_name,
            "pipeline_warnings": warnings,
            "antipattern_warnings": antipattern_warnings,
            "misconception_alerts": misconceptions,
            "confidence": round(confidence * 100, 1),
            "confidence_label": confidence_parts["label"],
            "confidence_components": confidence_parts,
            "text": gen_output["text"],
            "code": gen_output.get("code", ""),
            "code_explained": explained_code,
            "code_output": code_result,
            "difficulty_level": difficulty,
            "mode": "direct",
            "hint": "",
            "why": why_data.get("why", ""),
            "when_to_use": why_data.get("when_to_use", ""),
            "common_pitfall": why_data.get("common_pitfall", ""),
            "suggested_questions": suggestions,
            "next_step": self.tracker.suggest_next_step(),
            "plan_subtasks": [s.description for s in plan.subtasks],
            "pedagogy_mode": plan.pedagogy_mode,
            "critic": {
                "factual_accuracy": critic_score.factual_accuracy,
                "pedagogical_fit": critic_score.pedagogical_fit,
                "completeness": critic_score.completeness,
                "feedback": critic_score.feedback,
                "retried": retried,
            },
            "retrieval_query": expanded_query,
        }
        self._log_event(session_id, result, int((time.perf_counter() - t0) * 1000))
        return result

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
                "active_context": self.memory.active_context,
            },
        )
