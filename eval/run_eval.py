"""
DS Mentor Pro — Evaluation framework.
Metrics: stage accuracy, confidence, WHY coverage, suggestion quality,
Socratic activation rate, learning progression.
"""
import csv
import json
import os

from core.agent import MentorAgent
from core.socratic_engine import SocraticEngine


def _load_rows():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "data", "dataset.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_eval():
    """Core evaluation: stage accuracy + average confidence."""
    rows = _load_rows()
    if not rows:
        return {"error": "Missing dataset.csv"}

    agent = MentorAgent(data_docs=rows)
    total = min(25, len(rows))
    correct_stage = 0
    confidence_sum = 0.0

    for row in rows[:total]:
        result = agent.process_sync(query=row["query"])
        if int(row.get("stage", 1)) == result["stage_num"]:
            correct_stage += 1
        confidence_sum += float(result["confidence"])

    return {
        "samples": total,
        "pipeline_stage_accuracy": round(correct_stage / total, 3),
        "avg_confidence_percent": round(confidence_sum / total, 2),
    }


def eval_why_coverage(rows=None):
    """What % of QA pairs have non-empty why_explanation?"""
    rows = rows or _load_rows()
    if not rows:
        return 0.0
    filled = sum(1 for r in rows if r.get("why_explanation", "").strip())
    return round(filled / len(rows), 3)


def eval_suggestion_quality(rows=None):
    """
    For each row's related_questions, check:
      - groundedness: do they exist in the KB?
      - diversity: are they different from the original question?
    """
    rows = rows or _load_rows()
    if not rows:
        return {"groundedness": 0.0, "diversity": 0.0}

    all_queries = {r["query"].lower().strip() for r in rows}
    grounded = 0
    diverse = 0
    total_suggestions = 0

    for r in rows:
        try:
            rqs = json.loads(r.get("related_questions", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue
        for rq in rqs:
            total_suggestions += 1
            if rq.lower().strip() in all_queries:
                grounded += 1
            if rq.lower().strip() != r["query"].lower().strip():
                diverse += 1

    if total_suggestions == 0:
        return {"groundedness": 0.0, "diversity": 0.0}

    return {
        "groundedness": round(grounded / total_suggestions, 3),
        "diversity": round(diverse / total_suggestions, 3),
    }


def eval_socratic_activation_rate(rows=None, skill_override: float = 0.5):
    """
    Of all conceptual queries, what fraction would trigger Socratic mode?
    Target: 15-25%.
    """
    rows = rows or _load_rows()
    engine = SocraticEngine()
    engine.interaction_counter["eval"] = engine.socratic_cooldown

    conceptual_count = 0
    activated = 0

    for r in rows:
        q = r.get("query", "")
        stage = int(r.get("stage", 1))
        if engine._is_conceptual_query(q):
            conceptual_count += 1
            if engine.should_activate(q, stage, skill_override, "eval"):
                activated += 1
                engine.record_interaction("eval", was_socratic=True)
            else:
                engine.record_interaction("eval", was_socratic=False)

    if conceptual_count == 0:
        return {"conceptual_queries": 0, "activation_rate": 0.0}

    return {
        "conceptual_queries": conceptual_count,
        "activated": activated,
        "activation_rate": round(activated / conceptual_count, 3),
    }


def eval_learning_progression(rows=None):
    """
    Check whether dataset difficulty increases across stages.
    Measure: average difficulty score of first-half vs second-half stages.
    """
    rows = rows or _load_rows()
    diff_map = {"beginner": 1, "intermediate": 2, "advanced": 3}

    by_stage = {}
    for r in rows:
        s = int(r.get("stage", 1))
        d = diff_map.get(r.get("difficulty", "intermediate"), 2)
        by_stage.setdefault(s, []).append(d)

    stage_avgs = {}
    for s in sorted(by_stage):
        vals = by_stage[s]
        stage_avgs[s] = round(sum(vals) / len(vals), 2)

    early = [stage_avgs.get(s, 2) for s in range(1, 4)]
    late = [stage_avgs.get(s, 2) for s in range(5, 8)]

    return {
        "stage_avg_difficulty": stage_avgs,
        "early_avg": round(sum(early) / len(early), 2) if early else 0,
        "late_avg": round(sum(late) / len(late), 2) if late else 0,
    }


if __name__ == "__main__":
    print("=== Core Eval ===")
    print(run_eval())
    print("\n=== WHY Coverage ===")
    print(eval_why_coverage())
    print("\n=== Suggestion Quality ===")
    print(eval_suggestion_quality())
    print("\n=== Socratic Activation ===")
    print(eval_socratic_activation_rate())
    print("\n=== Learning Progression ===")
    print(eval_learning_progression())
