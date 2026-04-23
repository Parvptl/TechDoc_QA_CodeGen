# DS Mentor Pro (Non-Backend README)

This document summarizes the parts of the codebase used outside backend/API services.
It focuses on models, data, evaluation, UI, and supporting modules.

## Scope

- Included: `models/`, `data/`, `core/`, `modules/`, `rag/`, `storage/`, `evaluation/`, `scripts/`, `ui/`, `tests/`
- Excluded intentionally: `services/` and backend route/server wiring

## Primary Models Used

- **Stage classifier**
  - File: `models/stage_classifier.py`
  - Main options:
    - DistilBERT classifier (`distilbert-base-uncased`) for 7-stage prediction
    - TF-IDF + `LinearSVC` (+ calibration) fallback
  - Outputs stage labels such as `Stage 4 - Preprocessing`

- **Code generation model**
  - File: `models/finetune_codet5.py`
  - Base model: `Salesforce/codet5-small`
  - Fine-tuning target: stage-aware DS code generation
  - Inference fallback chain:
    - fine-tuned CodeT5 (`models/codet5_finetuned`)
    - base CodeT5
    - template generator (`modules/code_generator.py`)

## Data and Dataset Pipelines

- **Dataset build**
  - File: `data/build_dataset.py`
  - Combines:
    - Kaggle notebook extraction (when enabled)
    - curated fallback data
  - Produces:
    - `dataset.csv`
    - `small_sample_dataset.csv`
    - `train.csv`, `val.csv`, `test.csv`

- **Stage labeling**
  - Files: `data/label_stages.py`, `scripts/run_strict_results_pipeline.py`
  - Produces stage-labeled train/val/test CSVs used by model training and evaluation

- **Runtime dataset preparation**
  - File: `scripts/prepare_runtime_dataset.py`
  - Converts split/full datasets to runtime-friendly format (`data/runtime_dataset.csv`)

## Non-Backend Core Intelligence Modules

- **Orchestration and tutoring logic**
  - `core/agent.py`
  - `core/project_mode.py`
  - `core/pipeline_tracker.py`

- **Learning support**
  - `core/socratic_engine.py`
  - `core/quiz_engine.py`
  - `core/question_generator.py`
  - `core/checkpoint_assessor.py`
  - `core/skill_assessor.py`

- **Code and quality assistance**
  - `core/code_engine.py`
  - `core/code_annotator.py`
  - `core/antipattern_detector.py`
  - `core/why_engine.py`

- **Retrieval and memory**
  - `core/retriever.py`
  - `core/retrieval/query_expand.py`
  - `core/memory.py`
  - `rag/hybrid_retriever.py`

## Supporting Modules and Storage

- **Modules**
  - `modules/retrieval.py`
  - `modules/workflow.py`
  - `modules/conversation.py`
  - `modules/confidence_scorer.py`
  - `modules/visualization.py`
  - `modules/visualization_sandbox.py`

- **Storage and reporting**
  - `storage/session_store.py`
  - `storage/vector_store.py`
  - `storage/learning_analytics.py`
  - `storage/report_exporter.py`

## Evaluation and Benchmarking

- **Metrics**
  - File: `evaluation/metrics.py`
  - Includes retrieval and text/code quality metrics (for example BLEU-1 and lightweight CodeBLEU-like score)

- **Benchmark suite**
  - File: `evaluation/benchmark.py`
  - Compares DS Mentor against baseline references across multi-metric QA/code tasks

- **Strict reproducible pipeline**
  - File: `scripts/run_strict_results_pipeline.py`
  - Typical outputs:
    - `outputs/stage_split_eval.json`
    - `outputs/codet5_split_eval.json`
    - `outputs/codet5_benchmark_eval.json`

## UI (Non-Backend)

- **Streamlit interfaces**
  - `ui/streamlit_app.py` (primary UI)
  - `ui/app.py` (alternate UI entry)

## Key Dependencies Used

From `requirements.txt`:

- **ML/NLP**: `torch`, `transformers`, `sentence-transformers`, `datasets`, `tokenizers`
- **Classical ML**: `scikit-learn`, `numpy`, `pandas`
- **Retrieval/NLP tooling**: `rank-bm25`, `spacy`, `nltk`
- **UI**: `streamlit`, `gradio`
- **Evaluation**: `rouge-score`, `sacrebleu`
- **Testing/Dev**: `pytest`, `pytest-cov`, `jupyter`, `loguru`

## Typical Non-Backend Workflow

1. Build/refresh dataset in `data/` (`build_dataset.py`)
2. Prepare runtime dataset (`scripts/prepare_runtime_dataset.py`)
3. Train/evaluate stage classifier and optional CodeT5 (`scripts/run_strict_results_pipeline.py`)
4. Run benchmark reports (`evaluation/benchmark.py` or strict pipeline outputs)
5. Use Streamlit UI (`ui/streamlit_app.py`) for interactive tutoring flow

