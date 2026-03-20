"""
create_dataset_v2.py — Extended dataset builder
Merges base curated set + extra advanced examples → 500+ total QA pairs.
Run this INSTEAD of create_dataset.py if you want the full dataset.
"""
import csv, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
random.seed(42)

from data.create_dataset import build_curated, augment, try_hf, STAGE_NAMES, STAGE_KEYWORDS
from data.dataset_extra import EXTRA_CURATED


def build_extended_dataset(out="data/dataset.csv"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Base curated set (140 unique + ~135 augmented)
    base_rows = build_curated()

    # Extra advanced examples
    extra_rows = []
    for stage, examples in EXTRA_CURATED.items():
        for explanation, code, difficulty in examples:
            extra_rows.append({
                "explanation": explanation,
                "code": code,
                "pipeline_stage": stage,
                "source": "curated_advanced",
                "difficulty": difficulty,
            })

    # Augment extra rows too
    extra_aug = augment(extra_rows[:])

    # Combine all
    all_rows = base_rows + extra_rows + extra_aug

    # Try HuggingFace supplement
    all_rows = try_hf(all_rows)

    random.shuffle(all_rows)

    # Save
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["explanation","code","pipeline_stage","source","difficulty"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[INFO] Extended dataset: {len(all_rows)} examples → {out}")
    for s in range(1, 8):
        n = sum(1 for r in all_rows if str(r["pipeline_stage"]) == str(s))
        print(f"  Stage {s} ({STAGE_NAMES[s]}): {n} examples")

    return all_rows


if __name__ == "__main__":
    build_extended_dataset()
