"""
Unified evaluation suite for DS Mentor Pro.

This complements (does not replace) existing scripts:
- evaluate.py
- eval/run_eval.py
- evaluation/benchmark.py

It adds:
- Precision@k, Recall@k, MRR, nDCG for retrieval with explicit relevance sets
- Classification metrics incl confusion matrix for stage classification
- Anti-pattern detection precision/recall/F1 against labeled snippets
- Response quality (BLEU-1 + ROUGE-L) vs reference answers

Run:
  python -m evaluation.run_suite --dataset evaluation/datasets/small_eval.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.metrics import (
    PRF1,
    bleu1,
    confusion_matrix_counts,
    ndcg_at_k,
    precision_at_k,
    prf1_from_counts,
    recall_at_k,
    reciprocal_rank,
    rouge_l,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_kb_dataset_csv() -> Path:
    """Ensure `data/dataset.csv` exists by generating it if missing."""
    root = _repo_root()
    csv_path = root / "data" / "dataset.csv"
    if csv_path.exists():
        return csv_path

    gen = root / "data" / "generate_dataset_simple.py"
    if not gen.exists():
        raise FileNotFoundError("Missing knowledge base and generator: data/dataset.csv and data/generate_dataset_simple.py")

    # Import and run generator (stdlib).
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_dataset_simple", str(gen))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.build_dataset()  # type: ignore[attr-defined]
    if not csv_path.exists():
        raise RuntimeError("Dataset generation ran but data/dataset.csv not created.")
    return csv_path


def _load_kb_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _load_eval_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _kb_query_to_index(rows: Sequence[dict]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, r in enumerate(rows):
        q = str(r.get("query", "")).strip()
        if q and q not in m:
            m[q] = i
    return m


# ---------------------------------------------------------------------------
# Anti-pattern mapping (strings -> stable ids)
# ---------------------------------------------------------------------------


ANTIPATTERN_ID_RULES: List[Tuple[str, List[str]]] = [
    ("data_leakage_fit_transform_before_split", ["data leakage: calling fit_transform", "fit_transform before train_test_split"]),
    ("predict_on_train_only", ["overfitting risk: predicting on training data"]),
    ("blind_dropna", ["indiscriminate dropna()", "dropna"]),
    ("no_validation_strategy", ["no validation strategy detected"]),
    ("accuracy_imbalanced_warning", ["accuracy can be misleading on imbalanced data"]),
    ("fit_on_test_set", ["fitting a model or transformer on the test set"]),
    ("feature_selection_before_split", ["feature selection before train_test_split"]),
]


def _normalize_antipattern_warnings(warnings: Sequence[str]) -> List[str]:
    out: List[str] = []
    for w in warnings or []:
        lw = str(w).lower().strip()
        for ap_id, needles in ANTIPATTERN_ID_RULES:
            if any(n in lw for n in needles):
                out.append(ap_id)
                break
    # stable unique list
    uniq: List[str] = []
    for x in out:
        if x not in uniq:
            uniq.append(x)
    return uniq


# ---------------------------------------------------------------------------
# Core evaluators
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float


def eval_retrieval(rows: Sequence[dict], eval_items: Sequence[dict]) -> RetrievalMetrics:
    from core.retriever import HybridRetriever as CoreHybridRetriever

    q2i = _kb_query_to_index(rows)
    retriever = CoreHybridRetriever(use_dense=False, use_cross_encoder=False)
    retriever.add_documents(list(rows))

    p1: List[float] = []
    p3: List[float] = []
    p5: List[float] = []
    r1: List[float] = []
    r3: List[float] = []
    r5: List[float] = []
    rrs: List[float] = []
    ndcgs: List[float] = []

    for ex in eval_items:
        rel_queries = [str(x) for x in ex.get("relevant_doc_queries", [])]
        # Ground truth relevance set (exact match by KB query string).
        rel_query_set = {q for q in rel_queries if q in q2i}
        if not rel_query_set:
            # Skip examples that cannot be grounded to KB entries.
            continue

        stage = int(ex.get("expected_stage", -1))
        retrieved = retriever.retrieve(
            query=str(ex.get("query", "")),
            active_stage=stage,
            top_k=5,
            skill_level=0.5,
        )
        # We treat a retrieved doc as relevant if its KB query string is labeled relevant.
        relevances = [1 if str(d.get("query", "")).strip() in rel_query_set else 0 for d in retrieved]
        n_rel = len(rel_query_set)

        p1.append(precision_at_k(relevances, 1))
        p3.append(precision_at_k(relevances, 3))
        p5.append(precision_at_k(relevances, 5))
        r1.append(recall_at_k(relevances, 1, n_rel))
        r3.append(recall_at_k(relevances, 3, n_rel))
        r5.append(recall_at_k(relevances, 5, n_rel))
        rrs.append(reciprocal_rank(relevances))
        ndcgs.append(ndcg_at_k(relevances, 5, n_rel))

    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return RetrievalMetrics(
        precision_at_1=round(mean(p1), 4),
        precision_at_3=round(mean(p3), 4),
        precision_at_5=round(mean(p5), 4),
        recall_at_1=round(mean(r1), 4),
        recall_at_3=round(mean(r3), 4),
        recall_at_5=round(mean(r5), 4),
        mrr=round(mean(rrs), 4),
        ndcg_at_5=round(mean(ndcgs), 4),
    )


@dataclass
class StageClfMetrics:
    accuracy: float
    macro_f1: float
    labels: List[int]
    confusion: List[List[int]]


def eval_stage_classification(eval_items: Sequence[dict]) -> StageClfMetrics:
    from core.stage_classifier import StageClassifier

    clf = StageClassifier()
    y_true: List[int] = []
    y_pred: List[int] = []
    for ex in eval_items:
        if "expected_stage" not in ex:
            continue
        yt = int(ex["expected_stage"])
        yp, _conf = clf.classify(str(ex.get("query", "")))
        y_true.append(yt)
        y_pred.append(int(yp))

    # Use sklearn if available (already a dependency), else fallback.
    try:
        from sklearn.metrics import accuracy_score, f1_score

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        acc = correct / float(max(1, len(y_true)))
        macro_f1 = 0.0

    labels, cm = confusion_matrix_counts(y_true, y_pred, labels=list(range(1, 8)))
    return StageClfMetrics(
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        labels=labels,
        confusion=cm,
    )


@dataclass
class AntiPatternMetrics:
    micro: PRF1
    per_id: Dict[str, PRF1]


def eval_antipattern_detection(eval_items: Sequence[dict]) -> AntiPatternMetrics:
    from core.antipattern_detector import AntiPatternDetector

    det = AntiPatternDetector()
    ids = [ap_id for ap_id, _needles in ANTIPATTERN_ID_RULES]

    # micro counts across ids
    tp = fp = fn = 0
    per: Dict[str, Tuple[int, int, int]] = {i: (0, 0, 0) for i in ids}  # tp, fp, fn

    for ex in eval_items:
        expected = set(ex.get("expected_antipattern_ids", []) or [])
        code = str(ex.get("provided_code", "") or "")
        if not expected and not code.strip():
            continue

        pred_warnings = det.check_code(code) if code.strip() else []
        pred = set(_normalize_antipattern_warnings(pred_warnings))

        for ap_id in ids:
            e = ap_id in expected
            p = ap_id in pred
            if p and e:
                tp += 1
                a, b, c = per[ap_id]
                per[ap_id] = (a + 1, b, c)
            elif p and not e:
                fp += 1
                a, b, c = per[ap_id]
                per[ap_id] = (a, b + 1, c)
            elif (not p) and e:
                fn += 1
                a, b, c = per[ap_id]
                per[ap_id] = (a, b, c + 1)

    per_scores: Dict[str, PRF1] = {k: prf1_from_counts(*v) for k, v in per.items()}
    return AntiPatternMetrics(micro=prf1_from_counts(tp, fp, fn), per_id=per_scores)


@dataclass
class ResponseQualityMetrics:
    bleu1: float
    rouge_l: float


def eval_response_quality(kb_rows: Sequence[dict], eval_items: Sequence[dict]) -> ResponseQualityMetrics:
    """Compare generated answer text vs reference KB answer for grounded examples."""
    from core.agent import MentorAgent

    agent = MentorAgent(data_docs=list(kb_rows))
    kb_by_query: Dict[str, dict] = {str(r.get("query", "")).strip(): r for r in kb_rows if str(r.get("query", "")).strip()}

    bleu_scores: List[float] = []
    rouge_scores: List[float] = []

    def extract_answer_section(text: str) -> str:
        """Best-effort extraction of the '**Answer**' section from the template generator."""
        t = str(text or "")
        anchor = "**Answer**"
        i = t.find(anchor)
        if i < 0:
            return t
        t2 = t[i + len(anchor) :]
        # Strip leading whitespace/newlines
        t2 = t2.lstrip()
        # Stop at the next section header.
        next_hdr = t2.find("\n**")
        if next_hdr >= 0:
            return t2[:next_hdr].strip()
        return t2.strip()

    for ex in eval_items:
        rel_qs = [str(x) for x in ex.get("relevant_doc_queries", [])]
        if len(rel_qs) != 1:
            # For response quality we want single-reference examples.
            continue
        ref_row = kb_by_query.get(rel_qs[0])
        if not ref_row:
            continue
        reference = str(ref_row.get("answer", "") or "")
        if not reference.strip():
            continue
        out = agent.process_sync(query=str(ex.get("query", "")), provided_code=str(ex.get("provided_code", "")))
        hyp = extract_answer_section(str(out.get("text", "") or ""))
        bleu_scores.append(bleu1(reference, hyp))
        rouge_scores.append(rouge_l(reference, hyp))

    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return ResponseQualityMetrics(bleu1=round(mean(bleu_scores), 4), rouge_l=round(mean(rouge_scores), 4))


# ---------------------------------------------------------------------------
# CLI + reporting
# ---------------------------------------------------------------------------


def _format_cm(labels: Sequence[int], cm: Sequence[Sequence[int]]) -> str:
    # Small markdown table.
    header = "| true\\pred | " + " | ".join(str(x) for x in labels) + " |\n"
    sep = "|---|" + "|".join(["---"] * len(labels)) + "|\n"
    rows = []
    for lab, row in zip(labels, cm):
        rows.append("| " + str(lab) + " | " + " | ".join(str(int(x)) for x in row) + " |")
    return header + sep + "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="evaluation/datasets/small_eval.jsonl", help="Path to eval jsonl")
    p.add_argument("--out_dir", default="outputs", help="Directory to write reports")
    args = p.parse_args(argv)

    csv_path = _ensure_kb_dataset_csv()
    kb_rows = _load_kb_rows(csv_path)
    eval_items = _load_eval_jsonl(_repo_root() / args.dataset)

    ret = eval_retrieval(kb_rows, eval_items)
    clf = eval_stage_classification(eval_items)
    ap = eval_antipattern_detection(eval_items)
    rq = eval_response_quality(kb_rows, eval_items)

    out_dir = _repo_root() / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary = {
        "retrieval": ret.__dict__,
        "stage_classification": {"accuracy": clf.accuracy, "macro_f1": clf.macro_f1},
        "antipattern_detection": {
            "micro": {"precision": ap.micro.precision, "recall": ap.micro.recall, "f1": ap.micro.f1},
            "per_id": {k: v.__dict__ for k, v in ap.per_id.items()},
        },
        "response_quality": rq.__dict__,
    }
    (out_dir / "eval_suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown report
    md = []
    md.append("## DS Mentor Pro — Evaluation Suite\n\n")
    md.append("### Retrieval\n\n")
    md.append("| Metric | Value |\n|---|---:|\n")
    md.append(f"| Precision@1 | {ret.precision_at_1:.4f} |\n")
    md.append(f"| Precision@3 | {ret.precision_at_3:.4f} |\n")
    md.append(f"| Precision@5 | {ret.precision_at_5:.4f} |\n")
    md.append(f"| Recall@1 | {ret.recall_at_1:.4f} |\n")
    md.append(f"| Recall@3 | {ret.recall_at_3:.4f} |\n")
    md.append(f"| Recall@5 | {ret.recall_at_5:.4f} |\n")
    md.append(f"| MRR | {ret.mrr:.4f} |\n")
    md.append(f"| nDCG@5 | {ret.ndcg_at_5:.4f} |\n\n")

    md.append("### Stage classification\n\n")
    md.append("| Metric | Value |\n|---|---:|\n")
    md.append(f"| Accuracy | {clf.accuracy:.4f} |\n")
    md.append(f"| Macro-F1 | {clf.macro_f1:.4f} |\n\n")
    md.append("**Confusion matrix (stages 1–7)**\n\n")
    md.append(_format_cm(clf.labels, clf.confusion))
    md.append("\n\n")

    md.append("### Anti-pattern detection\n\n")
    md.append("| Metric | Value |\n|---|---:|\n")
    md.append(f"| Precision (micro) | {ap.micro.precision:.4f} |\n")
    md.append(f"| Recall (micro) | {ap.micro.recall:.4f} |\n")
    md.append(f"| F1 (micro) | {ap.micro.f1:.4f} |\n\n")
    md.append("Per-pattern PRF1:\n\n")
    md.append("| Pattern id | Precision | Recall | F1 |\n|---|---:|---:|---:|\n")
    for k in sorted(ap.per_id):
        v = ap.per_id[k]
        md.append(f"| `{k}` | {v.precision:.4f} | {v.recall:.4f} | {v.f1:.4f} |\n")
    md.append("\n")

    md.append("### Response quality\n\n")
    md.append("| Metric | Value |\n|---|---:|\n")
    md.append(f"| BLEU-1 | {rq.bleu1:.4f} |\n")
    md.append(f"| ROUGE-L | {rq.rouge_l:.4f} |\n")
    md.append("\n")

    (out_dir / "eval_suite_report.md").write_text("".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

