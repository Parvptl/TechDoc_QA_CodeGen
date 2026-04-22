# DS Mentor Pro

Beginner-focused NLP mentor for the complete data science workflow.

DS Mentor is designed to guide learners stage-by-stage (problem framing -> data loading -> EDA -> preprocessing -> feature engineering -> modeling -> evaluation). It combines retrieval-based grounding, stage classification, anti-pattern detection, and optional CodeT5 code generation.

---

## 1) What This Project Does

When a learner asks a question, DS Mentor:

1. Classifies the query into one of 7 data science stages.
2. Retrieves relevant grounded context from a curated knowledge base.
3. Generates a structured teaching response (answer, why, pitfalls, code).
4. Detects anti-patterns in provided code (e.g., data leakage).
5. Tracks session progress and suggests next logical learning steps.

This project prioritizes teaching quality and safe guidance over unconstrained free-form generation.

---

## 2) Repository Overview

Key directories:

- `core/` - primary runtime engine (agent orchestration, retriever, classifier, generator, detector, memory)
- `services/` - FastAPI app and routes
- `ui/` - Streamlit apps
- `data/` - dataset generators, expanded datasets, utilities
- `models/` - model training and inference utilities (stage classifier, CodeT5)
- `evaluation/` and `eval/` - evaluation pipelines and metrics
- `scripts/` - reproducible training/evaluation prep pipelines
- `outputs/` - generated reports and metric summaries
- `tests/` - unit/integration/smoke tests

---

## 3) Core Runtime Architecture

Main runtime path:

- `services/api.py` -> API server
- `services/runtime.py` -> singleton agent loader
- `core/agent.py` -> orchestrates full query pipeline

Core components:

- `core/stage_classifier.py` - stage prediction (trained model if present, heuristic fallback)
- `core/retriever.py` - BM25-first hybrid retrieval
- `core/generator.py` - structured response generator (optional CodeT5 code path)
- `core/antipattern_detector.py` - rule-based anti-pattern detection
- `core/memory.py` + `storage/session_store.py` - conversation/session memory

Optional generation model:

- `models/finetune_codet5.py` - CodeT5 training + inference helper

---

## 4) Data and Knowledge Base

### Runtime knowledge base schema

Runtime expects:

- `query`
- `stage`
- `answer`
- `code`
- `why_explanation`
- `when_to_use`
- `common_pitfall`
- `related_questions`
- `difficulty`

### Expanded data pipeline

Expanded dataset builder:

- `data/build_dataset.py`

It can:

- pull high-vote Kaggle notebooks
- extract markdown-code pairs
- infer DS stage labels
- quality-filter + deduplicate
- create train/val/test splits

Generated outputs (example):

- `data/kaggle_expanded/dataset.csv`
- `data/kaggle_expanded/train.csv`
- `data/kaggle_expanded/val.csv`
- `data/kaggle_expanded/test.csv`

### Runtime normalization helper

- `scripts/prepare_runtime_dataset.py`

Converts expanded schema into runtime schema and writes:

- `data/runtime_dataset.csv`

---

## 5) Environment Variables

Important runtime flags:

- `DATASET_PATH` (default fallback chain: `data/runtime_dataset.csv` -> `data/dataset.csv`)
- `USE_CODET5` (`0` or `1`) controls whether generator attempts CodeT5 path
- `OPENAI_API_KEY` (optional, for optional WHY-engine LLM mode)

Deployment defaults in this repo now point to:

- `DATASET_PATH=data/runtime_dataset.csv`
- `USE_CODET5=1` (deployment uses CodeT5 path when model is available)

---

## 6) Installation and Quick Start (Classic)

From repo root:

```bash
pip install -r requirements.txt
python data/generate_dataset_simple.py
streamlit run ui/streamlit_app.py
```

Open:

- `http://localhost:8501`

Run tests:

```bash
python -m pytest tests/ -v
python tests/smoke_test.py
```

---

## 7) Strict Reproducible Results Pipeline (Recommended)

One command pipeline (Kaggle expansion + train/val/test training + eval):

```bash
python scripts/run_strict_results_pipeline.py --max-notebooks 20 --min-votes 30
```

Include CodeT5 smoke training/evaluation:

```bash
python scripts/run_strict_results_pipeline.py \
  --max-notebooks 20 \
  --min-votes 30 \
  --train-codet5 \
  --codet5-epochs 1 \
  --codet5-batch 2
```

What it does:

1. Builds expanded dataset from high-vote notebooks
2. Creates split-labeled files (`train/val/test`)
3. Prepares runtime dataset (`data/runtime_dataset.csv`)
4. Trains stage model on train split only
5. Evaluates stage model on train/val/test
6. Optional CodeT5 smoke train on train-only subset
7. Evaluates CodeT5 on held-out val/test

---

## 8) Training Workflows

### 8.1 Stage classifier

Train:

```bash
python -c "from models.stage_classifier import train_tfidf_svm; train_tfidf_svm(data_path='data/kaggle_expanded/stage_labeled_train.csv', save_path='models/tfidf_svm_fallback.pkl')"
```

### 8.2 CodeT5

Smoke train:

```bash
python -c "from models.finetune_codet5 import finetune_codet5; finetune_codet5(data_path='data/kaggle_expanded/stage_labeled_train_smoke.csv', output_dir='models/codet5_finetuned_train_smoke', epochs=1, batch_size=2)"
```

If Trainer complains about accelerate:

```bash
pip install accelerate
```

---

## 9) Evaluation Workflows

### 9.1 Unified suite

```bash
python -m evaluation.run_suite --dataset evaluation/datasets/small_eval.jsonl
```

Outputs:

- `outputs/eval_suite_summary.json`
- `outputs/eval_suite_report.md`

### 9.2 Strict split reports

Generated by strict script:

- `outputs/stage_split_eval.json` (stage train/val/test metrics)
- `outputs/codet5_split_eval.json` (CodeT5 held-out metrics)

### 9.3 Runtime dataset suite snapshot

Saved as:

- `outputs/eval_suite_summary_runtime.json`

---

## 10) Metrics Used and Why

### Retrieval metrics

- `Precision@k`, `Recall@k`, `MRR`, `nDCG@k`

Why:

- measure relevance coverage and ranking quality of retrieved context

### Classification metrics

- `Accuracy`, `Macro-F1`

Why:

- stage routing strongly affects final answer quality; Macro-F1 protects against stage imbalance

### Detection metrics

- micro `Precision/Recall/F1`

Why:

- anti-pattern detector should be precise and still catch important mistakes

### Generation/code metrics

- `BLEU-1`, `ROUGE-L`
- code `syntax_rate`, token overlap, line-match proxy

Why:

- text overlap and code validity/overlap are lightweight quality proxies

---

## 11) Deployment

### Render

`render.yaml` now:

- installs deploy dependencies
- generates runtime dataset
- prepares CodeT5 model directory
- starts FastAPI with `uvicorn`

### Docker

`Dockerfile` now:

- installs deploy dependencies
- generates runtime dataset
- prepares CodeT5 model directory
- sets:
  - `DATASET_PATH=data/runtime_dataset.csv`
  - `USE_CODET5=1`

### Deploy-time CodeT5 preparation

Script:

- `scripts/prepare_codet5_model.py`

Order:

1. use existing `models/codet5_finetuned` if present
2. else copy from `models/codet5_finetuned_train_smoke` if present
3. else download base `Salesforce/codet5-small` to target

---

## 12) API and UI Entrypoints

API:

```bash
uvicorn services.api:app --host 0.0.0.0 --port 8000
```

UI:

```bash
streamlit run ui/streamlit_app.py
```

---

## 13) Troubleshooting

### File lock errors on Windows (e.g., `data/dataset.csv`)

- use `data/runtime_dataset.csv` and `DATASET_PATH` instead of overwriting locked files

### CodeT5 trainer argument mismatch

- repo uses `eval_strategy` for transformers compatibility in this environment

### `accelerate` missing

```bash
pip install accelerate
```

### Kaggle pull issues

- verify Kaggle auth:

```bash
python -c "import kaggle; kaggle.api.authenticate(); print('ok')"
```

---

## 14) Current Project Positioning

This is a teaching-oriented DS mentor system:

- strong stage-wise guidance
- grounded retrieval
- safety checks for common learner mistakes
- optional neural code generation

Best use case:

- beginner to intermediate learners who need structured workflow support, not just raw answers.

---

## 15) Suggested Next Steps

- run full CodeT5 training (more epochs, larger train subset)
- add larger held-out benchmark suite for retrieval/generation
- add runtime telemetry for generation mode (`template` vs `codet5`)
- add citations for retrieved snippets in final response

