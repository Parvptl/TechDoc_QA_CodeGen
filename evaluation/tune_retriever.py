from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from rank_bm25 import BM25Okapi

from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.run_suite import _kb_query_to_index, _load_eval_jsonl, _load_kb_rows


@dataclass
class RetrievalResult:
    k1: float
    b: float
    stage_boost: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def eval_retrieval_bm25(rows: Sequence[dict], eval_items: Sequence[dict], k1: float, b: float, stage_boost: float) -> RetrievalResult:
    docs = list(rows)
    q2i = _kb_query_to_index(docs)
    tokenized = [str(d.get("query", "")).lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)

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
        rel_query_set = {q for q in rel_queries if q in q2i}
        if not rel_query_set:
            continue

        query = str(ex.get("query", ""))
        stage = int(ex.get("expected_stage", -1))
        scores = bm25.get_scores(query.lower().split())

        ranked = []
        for i, s in enumerate(scores):
            score = float(s)
            if stage > 0 and int(docs[i].get("stage", -1)) == stage:
                score *= stage_boost
            ranked.append((i, score))
        ranked.sort(key=lambda x: x[1], reverse=True)

        top = [docs[i] for i, _ in ranked[:5]]
        relevances = [1 if str(d.get("query", "")).strip() in rel_query_set else 0 for d in top]
        n_rel = len(rel_query_set)

        p1.append(precision_at_k(relevances, 1))
        p3.append(precision_at_k(relevances, 3))
        p5.append(precision_at_k(relevances, 5))
        r1.append(recall_at_k(relevances, 1, n_rel))
        r3.append(recall_at_k(relevances, 3, n_rel))
        r5.append(recall_at_k(relevances, 5, n_rel))
        rrs.append(reciprocal_rank(relevances))
        ndcgs.append(ndcg_at_k(relevances, 5, n_rel))

    return RetrievalResult(
        k1=k1,
        b=b,
        stage_boost=stage_boost,
        precision_at_1=round(_mean(p1), 4),
        precision_at_3=round(_mean(p3), 4),
        precision_at_5=round(_mean(p5), 4),
        recall_at_1=round(_mean(r1), 4),
        recall_at_3=round(_mean(r3), 4),
        recall_at_5=round(_mean(r5), 4),
        mrr=round(_mean(rrs), 4),
        ndcg_at_5=round(_mean(ndcgs), 4),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kb_csv", default="data/runtime_dataset.csv")
    p.add_argument("--dataset", default="evaluation/datasets/proxy_full_retrieval_eval.jsonl")
    p.add_argument("--out_json", default="outputs/retriever_tuning_full.json")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--grid", choices=["small", "full"], default="small")
    args = p.parse_args()

    rows = _load_kb_rows(Path(args.kb_csv))
    eval_items = _load_eval_jsonl(Path(args.dataset))

    if args.grid == "small":
        k1_vals = [1.0, 1.5, 2.0]
        b_vals = [0.5, 0.75, 0.9]
        stage_boost_vals = [1.0, 1.2, 1.3]
    else:
        k1_vals = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        b_vals = [0.3, 0.5, 0.7, 0.75, 0.9]
        stage_boost_vals = [1.0, 1.1, 1.2, 1.3, 1.5]

    results: List[RetrievalResult] = []
    for k1, b, sb in itertools.product(k1_vals, b_vals, stage_boost_vals):
        results.append(eval_retrieval_bm25(rows, eval_items, k1, b, sb))

    # prioritize first relevant rank quality, then ranking quality, then top1 precision
    results.sort(key=lambda x: (x.mrr, x.ndcg_at_5, x.precision_at_1), reverse=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")

    print(f"Grid tried: {len(results)} configs")
    print("Top configs:")
    for i, r in enumerate(results[: args.top_n], start=1):
        print(
            f"{i:02d} | k1={r.k1:.2f} b={r.b:.2f} boost={r.stage_boost:.2f} | "
            f"P@1={r.precision_at_1:.4f} R@5={r.recall_at_5:.4f} MRR={r.mrr:.4f} nDCG@5={r.ndcg_at_5:.4f}"
        )
    print(f"\nSaved full tuning results -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

