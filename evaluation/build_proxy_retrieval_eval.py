"""
Build a proxy retrieval eval JSONL from val/test CSV splits.

Why "proxy":
- We do not have manual retrieval relevance labels for each query.
- We approximate relevance by mapping each eval row to its own KB query text.

Output format matches evaluation.run_suite expectations:
{
  "query": "...",
  "expected_stage": 4,
  "relevant_doc_queries": ["..."]
}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def _build_kb_query_index(runtime_csv: Path) -> Dict[str, str]:
    """
    Build map from normalized query text to original query.
    Expects runtime CSV with column `query`.
    """
    idx: Dict[str, str] = {}
    rows = _load_rows(runtime_csv)
    for r in rows:
        q = str(r.get("query", ""))
        nq = _normalize_text(q).lower()
        if nq and nq not in idx:
            idx[nq] = q
    return idx


def build_proxy_eval(val_csv: Path, test_csv: Path, runtime_csv: Path, out_jsonl: Path) -> dict:
    kb_idx = _build_kb_query_index(runtime_csv)
    eval_rows = _load_rows(val_csv) + _load_rows(test_csv)

    items = []
    dropped = 0
    for r in eval_rows:
        # Option B proxy: use explanation as retrieval query.
        q = str(r.get("explanation", "")).strip()
        if not q:
            dropped += 1
            continue

        nq = _normalize_text(q).lower()
        kb_q = kb_idx.get(nq)
        if not kb_q:
            # If exact match not in runtime KB, skip.
            dropped += 1
            continue

        try:
            stage = int(r.get("pipeline_stage", 1))
        except (TypeError, ValueError):
            stage = 1

        items.append(
            {
                "query": q,
                "expected_stage": stage,
                "relevant_doc_queries": [kb_q],
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return {
        "written": len(items),
        "dropped": dropped,
        "source_rows": len(eval_rows),
        "out_path": str(out_jsonl),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", default="data/kaggle_expanded/val.csv")
    p.add_argument("--test_csv", default="data/kaggle_expanded/test.csv")
    p.add_argument("--runtime_csv", default="data/runtime_dataset.csv")
    p.add_argument("--out_jsonl", default="evaluation/datasets/proxy_full_retrieval_eval.jsonl")
    args = p.parse_args()

    stats = build_proxy_eval(
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv),
        runtime_csv=Path(args.runtime_csv),
        out_jsonl=Path(args.out_jsonl),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

