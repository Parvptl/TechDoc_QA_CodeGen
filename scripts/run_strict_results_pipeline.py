"""Run strict train/val/test pipeline for stage model and CodeT5 smoke.

Steps:
1) Build Kaggle-expanded dataset
2) Build stage-labeled train/val/test files
3) Prepare runtime dataset
4) Train stage model on train split
5) Evaluate stage model on train/val/test
6) Train CodeT5 smoke model on train-only subset
7) Evaluate CodeT5 on held-out val/test samples
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


STAGE_NAME = {
    1: "Problem Understanding",
    2: "Data Loading",
    3: "Exploratory Data Analysis",
    4: "Preprocessing",
    5: "Feature Engineering",
    6: "Modeling",
    7: "Evaluation",
}


def run_cmd(cmd: list[str]) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def make_stage_labeled_splits(base_dir: Path) -> None:
    fields = [
        "explanation",
        "code",
        "original_stage",
        "predicted_stage",
        "stage_name",
        "confidence",
        "label_method",
        "source",
        "difficulty",
    ]
    for split in ("train", "val", "test"):
        src = base_dir / f"{split}.csv"
        dst = base_dir / f"stage_labeled_{split}.csv"
        rows = list(csv.DictReader(src.open(encoding="utf-8")))
        out = []
        for r in rows:
            st = int(r.get("pipeline_stage", 1))
            out.append(
                {
                    "explanation": r.get("explanation", ""),
                    "code": r.get("code", ""),
                    "original_stage": st,
                    "predicted_stage": st,
                    "stage_name": STAGE_NAME.get(st, "Unknown"),
                    "confidence": 1.0,
                    "label_method": "kaggle_stage_infer",
                    "source": r.get("source", "kaggle_notebook"),
                    "difficulty": r.get("difficulty", "intermediate"),
                }
            )
        with dst.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        print(f"[INFO] Wrote {dst} ({len(out)} rows)")


def stage_eval(model_path: Path, base_dir: Path, out_path: Path) -> None:
    import pickle

    clf = pickle.load(model_path.open("rb"))
    out = {}
    for split in ("train", "val", "test"):
        rows = list(csv.DictReader((base_dir / f"stage_labeled_{split}.csv").open(encoding="utf-8")))
        x = [(r.get("explanation", "") + " " + r.get("code", "")) for r in rows]
        y = [int(r.get("predicted_stage", 1)) for r in rows]
        yp = [int(v) for v in clf.predict(x)]
        out[split] = {
            "n": len(rows),
            "accuracy": round(float(accuracy_score(y, yp)), 4),
            "macro_f1": round(float(f1_score(y, yp, average="macro", zero_division=0)), 4),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[INFO] Stage split report: {out_path}")
    print(json.dumps(out, indent=2))


def make_train_smoke_subset(base_dir: Path, per_stage: int = 120) -> Path:
    src = base_dir / "stage_labeled_train.csv"
    dst = base_dir / "stage_labeled_train_smoke.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    by_stage: dict[int, list[dict]] = {i: [] for i in range(1, 8)}
    for r in rows:
        by_stage[int(r.get("predicted_stage", 1))].append(r)
    random.seed(42)
    out = []
    for s in range(1, 8):
        random.shuffle(by_stage[s])
        out.extend(by_stage[s][:per_stage])
    random.shuffle(out)
    fields = [
        "explanation",
        "code",
        "original_stage",
        "predicted_stage",
        "stage_name",
        "confidence",
        "label_method",
        "source",
        "difficulty",
    ]
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"[INFO] Wrote train smoke subset: {dst} ({len(out)} rows)")
    return dst


def codet5_eval(base_dir: Path, out_path: Path, sample_n: int = 40) -> None:
    from models.finetune_codet5 import compute_code_metrics, generate_codet5

    report = {}
    random.seed(42)
    for split in ("val", "test"):
        rows = list(csv.DictReader((base_dir / f"stage_labeled_{split}.csv").open(encoding="utf-8")))
        sample = random.sample(rows, min(sample_n, len(rows)))
        syntax, tok, line = [], [], []
        methods = {}
        for r in sample:
            st = int(r.get("predicted_stage", 1))
            q = str(r.get("explanation", ""))[:220]
            out = generate_codet5(q, STAGE_NAME.get(st, ""), max_new_tokens=140)
            m = compute_code_metrics(str(out.get("code", "")), str(r.get("code", "")))
            syntax.append(1.0 if m["syntax_valid"] else 0.0)
            tok.append(float(m["token_overlap"]))
            line.append(float(m["line_match"]))
            key = str(out.get("method", "unknown"))
            methods[key] = methods.get(key, 0) + 1
        report[split] = {
            "n_eval": len(sample),
            "syntax_rate": round(sum(syntax) / max(1, len(syntax)), 4),
            "avg_token_overlap": round(sum(tok) / max(1, len(tok)), 4),
            "avg_line_match": round(sum(line) / max(1, len(line)), 4),
            "methods": methods,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] CodeT5 split report: {out_path}")
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-notebooks", type=int, default=20)
    parser.add_argument("--min-votes", type=int, default=30)
    parser.add_argument("--kaggle-output", default="data/kaggle_expanded")
    parser.add_argument("--train-codet5", action="store_true", help="Run CodeT5 smoke training/eval.")
    parser.add_argument("--codet5-epochs", type=int, default=1)
    parser.add_argument("--codet5-batch", type=int, default=2)
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_output)

    # 1) Build expanded dataset
    run_cmd(
        [
            sys.executable,
            "data/build_dataset.py",
            "--output-dir",
            str(kaggle_dir),
            "--max-notebooks",
            str(args.max_notebooks),
            "--min-votes",
            str(args.min_votes),
        ]
    )

    # 2) Stage-labeled split files
    make_stage_labeled_splits(kaggle_dir)

    # 3) Runtime dataset
    run_cmd(
        [
            sys.executable,
            "scripts/prepare_runtime_dataset.py",
            "--source",
            str(kaggle_dir / "dataset.csv"),
            "--target",
            "data/runtime_dataset.csv",
        ]
    )

    # 4) Train stage model (train only)
    run_cmd(
        [
            sys.executable,
            "-c",
            (
                "from models.stage_classifier import train_tfidf_svm; "
                "train_tfidf_svm(data_path='data/kaggle_expanded/stage_labeled_train.csv', "
                "save_path='models/tfidf_svm_fallback.pkl')"
            ),
        ]
    )

    # 5) Evaluate stage model (train/val/test)
    stage_eval(
        model_path=Path("models/tfidf_svm_fallback.pkl"),
        base_dir=kaggle_dir,
        out_path=Path("outputs/stage_split_eval.json"),
    )

    if args.train_codet5:
        # 6) Train CodeT5 smoke on train-only subset
        smoke_path = make_train_smoke_subset(kaggle_dir, per_stage=120)
        run_cmd(
            [
                sys.executable,
                "-c",
                (
                    "from models.finetune_codet5 import finetune_codet5; "
                    f"ok=finetune_codet5(data_path=r'{smoke_path.as_posix()}', "
                    "output_dir='models/codet5_finetuned_train_smoke', "
                    f"epochs={args.codet5_epochs}, batch_size={args.codet5_batch}); "
                    "print('train_ok',ok)"
                ),
            ]
        )

        # Activate this model for evaluation helper.
        active = Path("models/codet5_finetuned")
        if active.exists():
            shutil.rmtree(active)
        shutil.copytree(Path("models/codet5_finetuned_train_smoke"), active)

        # 7) Evaluate on held-out val/test
        codet5_eval(
            base_dir=kaggle_dir,
            out_path=Path("outputs/codet5_split_eval.json"),
            sample_n=40,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

