"""Checkpoint evidence scoring for project-based learning mode."""
from dataclasses import dataclass
from typing import Dict, List
import re


@dataclass
class CheckpointAssessment:
    score: float
    passed: bool
    feedback: str
    matched_keywords: List[str]
    min_required_score: float


class CheckpointAssessor:
    """Scores user evidence quality for stage checkpoints."""

    STAGE_KEYWORDS: Dict[int, List[str]] = {
        1: ["objective", "target", "metric", "baseline", "problem"],
        2: ["read_csv", "shape", "dtype", "columns", "load"],
        3: ["distribution", "correlation", "outlier", "missing", "plot"],
        4: ["impute", "fillna", "encode", "scale", "preprocess"],
        5: ["feature", "selection", "pca", "engineer", "transform"],
        6: ["train", "fit", "model", "cross_val", "random forest"],
        7: ["f1", "auc", "precision", "recall", "confusion"],
    }

    def __init__(self, min_required_score: float = 0.55):
        self.min_required_score = min_required_score

    def assess(self, stage_num: int, evidence: str) -> CheckpointAssessment:
        text = (evidence or "").strip()
        lower = text.lower()
        word_count = len(re.findall(r"[a-zA-Z0-9_]+", lower))

        # Basic quality features
        length_score = min(1.0, word_count / 30.0)
        code_score = 1.0 if any(tok in lower for tok in ("import ", "=", "def ", "plt.", "model.", "train_test_split")) else 0.0

        keywords = self.STAGE_KEYWORDS.get(int(stage_num), [])
        matched = [kw for kw in keywords if kw in lower]
        keyword_score = len(matched) / max(1, len(keywords))

        score = (0.60 * keyword_score) + (0.30 * length_score) + (0.10 * code_score)
        score = max(0.0, min(1.0, score))
        passed = score >= self.min_required_score
        if passed:
            feedback = "Checkpoint evidence is sufficient. Good stage-specific grounding."
        else:
            feedback = (
                "Evidence is too weak. Add stage-specific details (concepts/metrics) "
                "and optionally a small code snippet."
            )

        return CheckpointAssessment(
            score=round(score, 3),
            passed=passed,
            feedback=feedback,
            matched_keywords=matched,
            min_required_score=self.min_required_score,
        )
