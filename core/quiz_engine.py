"""Quiz generation and grading utilities."""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class QuizGrade:
    score: float
    correct: int
    total: int
    details: List[Dict[str, Any]]


class QuizEngine:
    """Generates and grades lightweight adaptive quizzes."""

    def generate(self, stage_name: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": "q1",
                "type": "mcq",
                "question": f"Which metric is usually better than accuracy for imbalanced {stage_name.lower()} tasks?",
                "options": ["Accuracy", "Precision/Recall/F1", "MSE", "R2"],
                "answer": "Precision/Recall/F1",
            },
            {
                "id": "q2",
                "type": "code_completion",
                "question": "Fill the blank: X_train, X_test, y_train, y_test = ______(X, y, test_size=0.2, random_state=42)",
                "answer": "train_test_split",
            },
            {
                "id": "q3",
                "type": "short_answer",
                "question": f"Name one common pitfall in {stage_name}.",
                "answer_keywords": ["leakage", "overfitting", "imbalance", "metric", "validation", "test set"],
            },
        ]

    def grade(self, questions: List[Dict[str, Any]], answers: List[Dict[str, str]]) -> QuizGrade:
        answer_map = {str(a.get("id")): str(a.get("answer", "")).strip().lower() for a in (answers or [])}
        details: List[Dict[str, Any]] = []
        correct = 0

        for q in questions:
            qid = str(q.get("id"))
            user_ans = answer_map.get(qid, "")
            is_correct = False
            if q.get("type") in ("mcq", "code_completion"):
                expected = str(q.get("answer", "")).strip().lower()
                is_correct = expected == user_ans
            elif q.get("type") == "short_answer":
                kws = [k.lower() for k in q.get("answer_keywords", [])]
                is_correct = any(kw in user_ans for kw in kws) and len(user_ans) >= 10

            if is_correct:
                correct += 1
            details.append({"id": qid, "correct": is_correct})

        total = max(1, len(questions))
        score = round(correct / total, 3)
        return QuizGrade(score=score, correct=correct, total=len(questions), details=details)
