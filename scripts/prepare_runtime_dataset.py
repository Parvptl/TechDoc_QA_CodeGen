"""Prepare runtime knowledge base CSV from expanded dataset schema.

Input schema (expanded):
  explanation, code, pipeline_stage, source, difficulty, ...

Output schema (runtime):
  query, stage, answer, code, why_explanation, when_to_use,
  common_pitfall, related_questions, difficulty
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNTIME_FIELDS = [
    "query",
    "stage",
    "answer",
    "code",
    "why_explanation",
    "when_to_use",
    "common_pitfall",
    "related_questions",
    "difficulty",
]


def build_runtime_rows(src_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in src_rows:
        explanation = str(r.get("explanation", "")).strip()
        code = str(r.get("code", "")).strip()
        if not explanation or not code:
            continue
        try:
            stage = int(r.get("pipeline_stage", r.get("stage", 1)))
        except (TypeError, ValueError):
            stage = 1

        query = explanation if len(explanation) <= 180 else (explanation[:180].rstrip() + "?")
        difficulty = str(r.get("difficulty", "intermediate") or "intermediate")
        rows.append(
            {
                "query": query,
                "stage": stage,
                "answer": explanation,
                "code": code,
                "why_explanation": (
                    f"This guidance is grounded in practical stage-{stage} data science workflows "
                    f"from curated and high-vote notebook patterns."
                ),
                "when_to_use": f"Use this while working on stage {stage} tasks in a tabular ML pipeline.",
                "common_pitfall": (
                    "Applying preprocessing or feature transforms before proper train/test split "
                    "can leak information and inflate performance."
                ),
                "related_questions": json.dumps([], ensure_ascii=True),
                "difficulty": difficulty,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/kaggle_expanded/dataset.csv")
    parser.add_argument("--target", default="data/runtime_dataset.csv")
    parser.add_argument(
        "--also-copy-to-dataset",
        action="store_true",
        help="Overwrite data/dataset.csv with the generated runtime dataset.",
    )
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.target)
    if not src.exists():
        raise FileNotFoundError(f"Missing source CSV: {src}")

    with src.open("r", newline="", encoding="utf-8") as f:
        src_rows = list(csv.DictReader(f))
    runtime_rows = build_runtime_rows(src_rows)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_FIELDS)
        writer.writeheader()
        writer.writerows(runtime_rows)

    print(f"[INFO] Wrote runtime dataset: {dst} ({len(runtime_rows)} rows)")

    if args.also_copy_to_dataset:
        data_dataset = Path("data/dataset.csv")
        data_dataset.write_text(dst.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[INFO] Copied runtime dataset to {data_dataset}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

