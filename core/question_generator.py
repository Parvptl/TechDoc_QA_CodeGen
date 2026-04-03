"""
Generates follow-up question suggestions after each answer.
Three suggestion types, selected based on user skill level:
- Lateral:    Same concept, different angle (for beginners)
- Deeper:     Same topic, more advanced (for intermediate)
- Connected:  Links to next pipeline stage (for advanced)
"""
import json
import re
from typing import List, Dict

from .stage_classifier import StageClassifier


class QuestionGenerator:
    """Suggests 2-3 follow-up questions to guide learning progression."""

    STAGE_NAMES = StageClassifier.STAGE_NAMES

    def __init__(self, knowledge_base: list = None):
        """
        Args:
            knowledge_base: loaded dataset rows with 'related_questions' field
        """
        self.kb = knowledge_base or []
        self._by_stage: Dict[int, List[dict]] = {}
        for doc in self.kb:
            try:
                s = int(doc.get("stage", 1))
            except (TypeError, ValueError):
                s = 1
            self._by_stage.setdefault(s, []).append(doc)

    def suggest(
        self,
        current_question: str,
        current_stage: int,
        skill_level: float,
        retrieved_docs: list,
        session_questions: list,
    ) -> List[dict]:
        """
        Generate 2-3 follow-up question suggestions.

        Selection logic based on skill:
          skill < 0.3:  2 lateral + 1 deeper
          0.3-0.7:      1 lateral + 1 deeper + 1 connected
          > 0.7:        1 deeper + 2 connected
        """
        if not self.kb and not retrieved_docs:
            return []

        pool = self._get_from_dataset(retrieved_docs)
        pool = self._filter_already_asked(pool, session_questions + [current_question])

        if skill_level < 0.3:
            target_mix = {"lateral": 2, "deeper": 1, "connected": 0}
        elif skill_level <= 0.7:
            target_mix = {"lateral": 1, "deeper": 1, "connected": 1}
        else:
            target_mix = {"lateral": 0, "deeper": 1, "connected": 2}

        classified = []
        for q in pool:
            qtype = self._classify_suggestion_type(q, current_stage)
            classified.append({"question": q, "type": qtype, "stage": self._infer_stage(q)})

        result = []
        for qtype, count in target_mix.items():
            matches = [c for c in classified if c["type"] == qtype and c not in result]
            result.extend(matches[:count])

        if len(result) < 2:
            extras = [c for c in classified if c not in result]
            result.extend(extras[: 3 - len(result)])

        return result[:3]

    def _get_from_dataset(self, retrieved_docs: list) -> List[str]:
        """Pull related_questions from retrieved documents + KB fallback."""
        pool: List[str] = []
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            raw = doc.get("related_questions", "[]")
            try:
                rqs = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                rqs = []
            if isinstance(rqs, list):
                pool.extend([q for q in rqs if isinstance(q, str) and q.strip()])

        if len(pool) < 3 and self.kb:
            for doc in self.kb[:20]:
                raw = doc.get("related_questions", "[]")
                try:
                    rqs = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(rqs, list):
                    pool.extend([q for q in rqs if isinstance(q, str) and q.strip()])

        seen = set()
        unique = []
        for q in pool:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return unique

    @staticmethod
    def _filter_already_asked(suggestions: list, session_questions: list) -> list:
        """Remove questions too similar to ones already asked."""
        asked_lower = {q.lower().strip() for q in session_questions}
        filtered = []
        for q in suggestions:
            q_norm = q.lower().strip()
            if q_norm not in asked_lower:
                overlap = sum(1 for a in asked_lower if _jaccard(q_norm, a) > 0.7)
                if overlap == 0:
                    filtered.append(q)
        return filtered

    def _classify_suggestion_type(self, suggestion: str, current_stage: int) -> str:
        """Classify a suggestion as lateral, deeper, or connected."""
        inferred_stage = self._infer_stage_num(suggestion)
        if inferred_stage != current_stage and inferred_stage > 0:
            return "connected"

        depth_signals = ["advanced", "compare", "difference", "vs", "tradeoff",
                         "when should", "is it better", "why does", "under the hood"]
        if any(s in suggestion.lower() for s in depth_signals):
            return "deeper"

        return "lateral"

    def _infer_stage(self, question: str) -> str:
        """Return stage name string for a question."""
        num = self._infer_stage_num(question)
        return self.STAGE_NAMES.get(num, "Problem Understanding")

    def _infer_stage_num(self, question: str) -> int:
        """Lightweight stage inference for a question string."""
        q = question.lower()
        stage_keywords = {
            1: ["objective", "goal", "target variable", "baseline", "problem", "metric"],
            2: ["load", "read", "csv", "json", "shape", "dtype"],
            3: ["distribution", "correlation", "outlier", "eda", "plot", "histogram", "missing pattern"],
            4: ["impute", "scale", "preprocess", "fillna", "encode", "clean", "missing value"],
            5: ["feature", "polynomial", "pca", "datetime", "one-hot", "interaction"],
            6: ["train", "model", "fit", "cross-validation", "hyperparameter", "random forest"],
            7: ["auc", "roc", "confusion", "f1", "evaluate", "learning curve", "shap", "precision"],
        }
        best, best_count = 1, 0
        for stage, kws in stage_keywords.items():
            count = sum(1 for kw in kws if kw in q)
            if count > best_count:
                best, best_count = stage, count
        return best


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    ta = set(re.findall(r"\w+", a))
    tb = set(re.findall(r"\w+", b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
