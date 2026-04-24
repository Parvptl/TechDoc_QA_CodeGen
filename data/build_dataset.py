"""Builds the DS Mentor dataset from Kaggle notebooks; falls back to curated data. Produces train/val/test splits."""

import csv
import json
import re
import random
import hashlib
import sys
from pathlib import Path
from typing import Optional, Tuple

# Subprocess / `python data/build_dataset.py` sets sys.path[0] to `data/`, not repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

random.seed(42)


# ── Stage definitions ─────────────────────────────────────────────────────────
STAGE_NAMES = {
    1: "Problem Understanding",
    2: "Data Loading",
    3: "Exploratory Data Analysis",
    4: "Preprocessing",
    5: "Feature Engineering",
    6: "Modeling",
    7: "Evaluation",
}

STAGE_KEYWORDS = {
    1: ["problem","objective","goal","business","define","task","competition",
        "target variable","predict","baseline","kpi","scope","success metric","null model"],
    2: ["read_csv","load_data","pd.read","import pandas","import numpy","load dataset",
        "read data","data loading","open file","glob","json.load","read_excel","read_sql",
        "database","parquet","encoding","chunk"],
    3: ["eda","exploratory","distribution","histogram","boxplot","countplot","correlation",
        "heatmap","pairplot","describe","value_counts","visualize","plot",
        "seaborn","matplotlib","skewness","kurtosis","nunique","outlier detection"],
    4: ["missing","null","fillna","dropna","impute","clean","outlier","duplicate",
        "strip","replace","dtype","astype","preprocessing","isnull","winsorize","clip",
        "smote","imbalance","regex"],
    5: ["feature","engineer","polynomial","interaction","encode","labelencoder","onehotencoder",
        "get_dummies","scaling","standardscaler","minmaxscaler","pca","tfidf","embedding",
        "log transform","target encode","frequency encode","datetime feature"],
    6: ["model","train","fit","predict","random forest","xgboost","lgbm","logistic regression",
        "svm","neural network","sklearn","cross_val","gridsearch","pipeline","classifier",
        "regressor","hyperparameter","optuna","stacking","ensemble"],
    7: ["accuracy","f1_score","roc_auc","precision","recall","confusion matrix",
        "classification_report","mse","rmse","mae","evaluate","metric","score",
        "learning curve","overfitting","underfitting","shap","feature importance","calibration"],
}

VISUAL_KEYWORDS = [
    "plt.show","plt.savefig","fig.show","sns.","plt.plot","plt.scatter",
    "plt.hist","plt.bar","plt.figure","plt.subplots","imshow","heatmap",
    "pairplot","violinplot","boxplot","histplot","barplot","countplot",
]

KAGGLE_COMPETITIONS = [
    "titanic", "house-prices-advanced-regression-techniques",
    "spaceship-titanic", "playground-series-s3e1",
    "store-sales-time-series-forecasting",
]


# ── Stage inference ────────────────────────────────────────────────────────────
def infer_stage(text: str) -> int:
    combined = text.lower()
    scores = {s: sum(1 for kw in kws if kw in combined)
              for s, kws in STAGE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 1


def has_visual_output(code: str) -> bool:
    return any(kw in code for kw in VISUAL_KEYWORDS)


# ── Deduplication ──────────────────────────────────────────────────────────────
def row_hash(explanation: str, code: str) -> str:
    return hashlib.md5((explanation[:100] + code[:100]).encode()).hexdigest()


def deduplicate(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        h = row_hash(row["explanation"], row["code"])
        if h not in seen:
            seen.add(h)
            unique.append(row)
    removed = len(rows) - len(unique)
    if removed:
        print(f"[INFO] Deduplication removed {removed} duplicates → {len(unique)} rows")
    return unique


# ── Quality filter ─────────────────────────────────────────────────────────────
def quality_filter(rows: list[dict]) -> list[dict]:
    filtered = []
    for row in rows:
        expl = row.get("explanation", "")
        code = row.get("code", "")
        # Minimum quality requirements
        if len(expl) < 15:
            continue
        if len(code) < 10:
            continue
        if len(code) > 3000:          # trim runaway cells
            row["code"] = code[:3000]
        if len(expl) > 1000:          # trim very long markdown
            row["explanation"] = expl[:1000]
        # Skip cells that are just imports or comments
        code_lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(code_lines) < 2:
            continue
        filtered.append(row)
    removed = len(rows) - len(filtered)
    if removed:
        print(f"[INFO] Quality filter removed {removed} rows → {len(filtered)} rows")
    return filtered


# ── Train / val / test split ───────────────────────────────────────────────────
def stratified_split(rows: list[dict], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Stratified split preserving stage distribution."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    from collections import defaultdict
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row["pipeline_stage"]].append(row)

    train, val, test = [], [], []
    for stage, stage_rows in by_stage.items():
        random.shuffle(stage_rows)
        n = len(stage_rows)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train.extend(stage_rows[:n_train])
        val.extend(stage_rows[n_train:n_train + n_val])
        test.extend(stage_rows[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"[INFO] Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ── Kaggle notebook extractor ──────────────────────────────────────────────────
def setup_kaggle():
    try:
        import kaggle
        kaggle.api.authenticate()
        return True
    except Exception as e:
        print(f"[INFO] Kaggle API unavailable ({type(e).__name__}). Using curated dataset.")
        return False


def _kernel_ref_and_votes(meta) -> Tuple[Optional[str], int]:
    """Normalize Kaggle kernel list entries across API / protobuf versions."""
    if meta is None:
        return None, 0
    if isinstance(meta, dict):
        ref = meta.get("ref")
        try:
            votes = int(meta.get("totalVotes", 0) or 0)
        except (TypeError, ValueError):
            votes = 0
        return (str(ref).strip() if ref else None), votes
    ref = getattr(meta, "ref", None)
    if ref is None:
        ref = getattr(meta, "Ref", None)
    votes_raw = getattr(meta, "totalVotes", None)
    if votes_raw is None:
        votes_raw = getattr(meta, "total_votes", None)
    try:
        votes = int(votes_raw or 0)
    except (TypeError, ValueError):
        votes = 0
    if ref:
        return str(ref).strip(), votes
    return None, votes


def _kernels_list_page(api, competition: str, page: int, page_size: int):
    """One page of kernels; supports older ``kaggle`` without ``page`` kwarg."""
    try:
        return api.kernels_list(
            competition=competition,
            page=page,
            page_size=page_size,
            language="python",
            kernel_type="notebook",
            sort_by="voteCount",
        ) or []
    except TypeError:
        if page > 1:
            return []
        return api.kernels_list(
            competition=competition,
            page_size=page_size,
            language="python",
            kernel_type="notebook",
            sort_by="voteCount",
        ) or []


def download_notebooks(
    competition: str,
    out_dir: Path,
    max_notebooks: int = 20,
    min_votes: int = 25,
) -> list[Path]:
    """
    List kernels via official Kaggle Python API, filter by votes, pull each .ipynb.

    Older project versions called ``kernels_list(..., output=dir)``; current
    ``kaggle`` packages return metadata only and require ``kernels_pull`` per kernel
    (same auth as ``setup_kaggle()`` — avoids flaky subprocess / CLI on Windows).
    """
    try:
        import kaggle

        api = kaggle.api
        out_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        page = 1
        page_size = 100  # server max for list_kernels

        while len(downloaded) < max_notebooks:
            kernels = _kernels_list_page(api, competition, page, page_size)

            if not kernels:
                break

            for meta in kernels:
                ref, votes = _kernel_ref_and_votes(meta)
                if not ref or votes < min_votes:
                    continue
                try:
                    api.kernels_pull(ref, path=str(out_dir), metadata=False, quiet=True)
                except Exception as pull_e:
                    print(f"[WARN] kernels_pull failed {ref}: {pull_e}")
                    continue
                stem = ref.split("/")[-1]
                nb_path = out_dir / f"{stem}.ipynb"
                if nb_path.exists():
                    downloaded.append(nb_path)
                if len(downloaded) >= max_notebooks:
                    break

            if len(downloaded) >= max_notebooks:
                break
            if len(kernels) < page_size:
                break
            page += 1
            if page > 50:
                print(f"[WARN] {competition}: stopped kernel list pagination at page {page}")
                break

        print(
            f"[INFO] {competition}: downloaded {len(downloaded)} notebooks "
            f"(votes>={min_votes}, cap={max_notebooks})"
        )
        return downloaded
    except Exception as e:
        print(f"[WARN] Failed to download {competition}: {e}")
        return []


def extract_pairs_from_notebook(nb_path: Path, competition: str) -> list[dict]:
    """Extract (markdown, code) pairs from a notebook with stage + visual labels."""
    pairs = []
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        return pairs

    cells = nb.get("cells", [])
    last_markdown = ""
    notebook_id = nb_path.stem

    for cell in cells:
        ct  = cell.get("cell_type", "")
        src = "".join(cell.get("source", []))

        if ct == "markdown":
            # Strip markdown formatting for cleaner text
            clean = re.sub(r"[#*_`]", "", src).strip()
            clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # [text](url) → text
            if len(clean) > 15:
                last_markdown = clean

        elif ct == "code" and last_markdown and len(src.strip()) > 10:
            combined = last_markdown + " " + src
            stage = infer_stage(combined)
            visual = has_visual_output(src)

            pairs.append({
                "explanation":    last_markdown,
                "code":           src.strip(),
                "pipeline_stage": stage,
                "has_visual":     visual,
                "source":         "kaggle_notebook",
                "difficulty":     "intermediate",
                "notebook_id":    notebook_id,
                "competition":    competition,
            })
            last_markdown = ""

    return pairs


def scrape_kaggle(competitions=None, max_notebooks_per_comp=20, min_votes: int = 25) -> list[dict]:
    """Main Kaggle scraping pipeline."""
    if not setup_kaggle():
        return []

    competitions = competitions or KAGGLE_COMPETITIONS
    all_pairs = []
    nb_dir = Path("data/raw_notebooks")

    for comp in competitions:
        comp_dir = nb_dir / comp
        notebooks = download_notebooks(comp, comp_dir, max_notebooks_per_comp, min_votes=min_votes)
        comp_pairs = 0
        for nb in notebooks:
            pairs = extract_pairs_from_notebook(nb, comp)
            all_pairs.extend(pairs)
            comp_pairs += len(pairs)
        print(f"[INFO] {comp}: extracted {comp_pairs} pairs, total so far={len(all_pairs)}")

    return all_pairs


# ── Curated fallback ───────────────────────────────────────────────────────────
def load_curated_fallback(source_csv: Optional[Path] = None) -> list[dict]:
    """Load existing curated dataset and add has_visual + notebook_id columns."""
    path = source_csv or Path("data/dataset.csv")
    if not path.exists():
        # Run create_dataset_v2 logic inline
        from data.create_dataset import build_curated
        from data.dataset_extra import EXTRA_CURATED
        rows = build_curated()
        for stage, examples in EXTRA_CURATED.items():
            for expl, code, diff in examples:
                rows.append({"explanation": expl, "code": code,
                             "pipeline_stage": stage, "source": "curated_advanced",
                             "difficulty": diff})
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    # Add missing columns
    for row in rows:
        # Support both legacy schemas:
        # 1) explanation/code/pipeline_stage
        # 2) query/answer/code/stage
        if "pipeline_stage" not in row:
            row["pipeline_stage"] = row.get("stage", 1)
        if "explanation" not in row:
            row["explanation"] = row.get("query", "")
        if "code" not in row:
            row["code"] = ""
        if "difficulty" not in row:
            row["difficulty"] = row.get("difficulty", "intermediate")
        if "has_visual" not in row:
            row["has_visual"] = has_visual_output(row.get("code", ""))
        if "notebook_id" not in row:
            row["notebook_id"] = "curated"
        if "competition" not in row:
            row["competition"] = "general_ds"
        # Ensure pipeline_stage is int-compatible
        row["pipeline_stage"] = int(row["pipeline_stage"])

    return rows


# ── Save helpers ───────────────────────────────────────────────────────────────
FIELDNAMES = ["explanation", "code", "pipeline_stage", "has_visual",
              "source", "difficulty", "notebook_id", "competition"]


def save_csv(rows: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved {len(rows)} rows → {path}")


# ── Main pipeline ──────────────────────────────────────────────────────────────
def build_full_dataset(
    use_kaggle: bool = True,
    output_dir: str = "data",
    competitions: Optional[list[str]] = None,
    max_notebooks: int = 20,
    min_votes: int = 25,
) -> dict:
    """
    Full dataset construction pipeline.
    Returns dict of {train, val, test, full} DataFrames.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Step 1: Collect data
    print("\n[STEP 1] Collecting data...")
    rows = []

    if use_kaggle:
        kaggle_rows = scrape_kaggle(
            competitions=competitions,
            max_notebooks_per_comp=max_notebooks,
            min_votes=min_votes,
        )
        if kaggle_rows:
            print(f"[INFO] Kaggle: {len(kaggle_rows)} raw pairs")
            rows.extend(kaggle_rows)

    # Always include curated data
    preferred_curated = Path(output_dir) / "dataset.csv"
    curated_source = preferred_curated if preferred_curated.exists() else Path("data/dataset.csv")
    curated = load_curated_fallback(curated_source)
    rows.extend(curated)
    print(f"[INFO] Total before cleaning: {len(rows)}")

    # Step 2: Clean and deduplicate
    print("\n[STEP 2] Cleaning and deduplicating...")
    rows = quality_filter(rows)
    rows = deduplicate(rows)

    # Step 3: Normalize pipeline_stage to int
    for row in rows:
        row["pipeline_stage"] = int(row["pipeline_stage"])
        row["has_visual"] = bool(row.get("has_visual", False))

    # Step 4: Print stats
    print(f"\n[INFO] Final dataset: {len(rows)} rows")
    from collections import Counter
    stage_counts = Counter(r["pipeline_stage"] for r in rows)
    for s in sorted(stage_counts):
        print(f"  Stage {s} ({STAGE_NAMES[s]}): {stage_counts[s]}")
    visual_count = sum(1 for r in rows if r["has_visual"])
    print(f"  Has visual: {visual_count} ({visual_count/len(rows)*100:.1f}%)")

    # Step 5: Save full dataset
    random.shuffle(rows)
    save_csv(rows, f"{output_dir}/dataset.csv")

    # Step 6: Generate sample
    sample = random.sample(rows, min(20, len(rows)))
    save_csv(sample, f"{output_dir}/small_sample_dataset.csv")

    # Step 7: Stratified split
    print("\n[STEP 3] Creating train/val/test splits...")
    train, val, test = stratified_split(rows)
    save_csv(train, f"{output_dir}/train.csv")
    save_csv(val,   f"{output_dir}/val.csv")
    save_csv(test,  f"{output_dir}/test.csv")

    return {"full": rows, "train": train, "val": val, "test": test}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DS Mentor Dataset Builder")
    parser.add_argument("--no-kaggle", action="store_true",
                        help="Skip Kaggle download, use curated data only")
    parser.add_argument("--competitions", nargs="+", default=None,
                        help="Kaggle competition slugs to download")
    parser.add_argument("--max-notebooks", type=int, default=20,
                        help="Max notebooks per competition")
    parser.add_argument("--min-votes", type=int, default=25,
                        help="Minimum upvotes required per notebook")
    parser.add_argument("--output-dir", default="data/kaggle_expanded",
                        help="Output directory for generated CSV files")
    args = parser.parse_args()

    result = build_full_dataset(
        use_kaggle=not args.no_kaggle,
        output_dir=args.output_dir,
        competitions=args.competitions,
        max_notebooks=args.max_notebooks,
        min_votes=args.min_votes,
    )
    print(f"\n✅ Dataset pipeline complete.")
    print(f"   Full:  {len(result['full'])} rows")
    print(f"   Train: {len(result['train'])} rows")
    print(f"   Val:   {len(result['val'])} rows")
    print(f"   Test:  {len(result['test'])} rows")
