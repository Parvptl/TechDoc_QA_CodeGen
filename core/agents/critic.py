"""Critic agent with retry gate for generated responses."""
from dataclasses import dataclass
from typing import List, Dict
import re


@dataclass
class CriticScore:
    factual_accuracy: float
    pedagogical_fit: float
    completeness: float
    feedback: str

    @property
    def min_axis(self) -> float:
        return min(self.factual_accuracy, self.pedagogical_fit, self.completeness)


class CriticAgent:
    """Evaluates response quality along factual/pedagogy/completeness axes."""

    def __init__(self, retry_threshold: float = 0.6):
        self.retry_threshold = retry_threshold

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()))

    def evaluate(
        self,
        response_text: str,
        retrieved_context: List[Dict],
        query: str,
        difficulty: str,
        required_subtasks: List[str] = None,
    ) -> CriticScore:
        response_tokens = self._tokenize(response_text)
        query_tokens = self._tokenize(query)

        context_blob = " ".join([str(d.get("answer", "")) for d in (retrieved_context or [])])
        context_tokens = self._tokenize(context_blob)

        # Factual support proxy: response terms grounded in retrieved terms.
        if response_tokens:
            factual = len(response_tokens & context_tokens) / max(1, len(response_tokens))
        else:
            factual = 0.0

        # Pedagogical fit proxy: beginner should include explanation cues,
        # advanced should stay concise and technical.
        low_diff = (difficulty or "").lower() == "beginner"
        has_explainers = any(tok in response_tokens for tok in ("because", "why", "example", "step"))
        if low_diff:
            pedagogy = 0.85 if has_explainers else 0.45
        else:
            pedagogy = 0.75

        # Completeness proxy: query coverage + required subtask mentions.
        query_overlap = len(query_tokens & response_tokens) / max(1, len(query_tokens))
        subtask_hits = 1.0
        if required_subtasks:
            required = set(t.lower() for t in required_subtasks)
            subtask_hits = len(required & response_tokens) / max(1, len(required))
        completeness = min(1.0, 0.7 * query_overlap + 0.3 * subtask_hits)

        feedback_parts = []
        if factual < self.retry_threshold:
            feedback_parts.append("Increase grounding in retrieved context.")
        if pedagogy < self.retry_threshold:
            feedback_parts.append("Adjust explanation depth to learner level.")
        if completeness < self.retry_threshold:
            feedback_parts.append("Cover missing parts of the user request.")
        feedback = " ".join(feedback_parts) if feedback_parts else "Response passes critic checks."

        return CriticScore(
            factual_accuracy=round(float(factual), 3),
            pedagogical_fit=round(float(pedagogy), 3),
            completeness=round(float(completeness), 3),
            feedback=feedback,
        )

    def should_retry(self, score: CriticScore) -> bool:
        return score.min_axis < self.retry_threshold
