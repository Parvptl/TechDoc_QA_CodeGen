# DS Mentor Pro -- Agentic Data Science Tutor

An agentic AI system that guides users through the complete data science pipeline using a ReAct-style agent, hybrid RAG retrieval, sandboxed code execution, and intelligent workflow tracking.

Built as a production-ready restructure of a course prototype. Fully modular, tested, and deployable.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **ReAct Agent Loop** | Sequential orchestrator: classify, retrieve, execute, detect, generate, score, persist. No LangChain dependency. |
| **Hybrid Retrieval** | BM25 (sparse) + optional dense embeddings + Reciprocal Rank Fusion + optional cross-encoder reranking. Stage-aware boosting. |
| **Sandboxed Code Execution** | AST-validated Python execution in isolated subprocess. Import whitelist, blocked dangerous operations, matplotlib capture. |
| **Pipeline Guardian** | Tracks progress through 7 DS stages. Detects skipped steps, suggests next steps. |
| **Anti-Pattern Detection** | Flags data leakage, missing train/test split, fitting on test data, blind dropna, no validation strategy, and more. |
| **Confidence Scoring** | Composite score from retrieval quality, classifier confidence, query-answer overlap, and code validity. |
| **Skill Adaptation** | Tracks user skill level per stage. Calibrates explanation depth (beginner / intermediate / advanced). |
| **Session Memory** | Multi-turn conversation with context preservation and persistence across messages. |
| **Dataset Profiler** | Upload CSV/Excel, auto-detect schema, compute summary statistics, inject dataset context into responses. |

---

## Architecture

```
                           +-----------------------+
                           |   Streamlit UI /       |
                           |   FastAPI Server        |
                           +-----------+-----------+
                                       |
                                       v
                           +-----------------------+
                           |   MentorAgent          |
                           |   (ReAct Orchestrator)  |
                           +-----------+-----------+
                                       |
           +---------------+-----------+-----------+----------------+
           v               v           v           v                v
    +-------------+ +----------+ +---------+ +----------+ +-------------+
    |   Stage     | | Hybrid   | |  Code   | |  Anti-   | |  Response   |
    | Classifier  | |Retriever | | Engine  | | Pattern  | | Generator   |
    | (7 stages)  | | BM25+RRF | | Sandbox | | Detector | | Template/LLM|
    +-------------+ +----------+ +---------+ +----------+ +-------------+
           |                                                      |
    +------+------------------------------------------------------+
    |
    +-- Pipeline Tracker (workflow state machine + next-step suggestions)
    +-- Confidence Scorer (composite reliability)
    +-- Session Memory (multi-turn context)
    +-- User Profile (skill tracking per stage)
    +-- Skill Assessor (adaptive difficulty)
    +-- Dataset Profiler (schema detection, summary, context injection)
```

### Agent Flow (per query)

```
 1. Classify  -> Identify pipeline stage (1-7)
 2. Socratic  -> Optionally respond with a guiding question instead of a direct answer
 3. Check     -> Detect skipped pipeline stages, generate warnings
 4. Retrieve  -> BM25 search over 109+ knowledge docs with stage-boosting
 5. Enrich    -> Extract WHY/WHEN/PITFALL from retrieved docs
 6. Profile   -> Inject dataset context if a dataset is uploaded
 7. Generate  -> Assemble structured response from context + code
 8. Execute   -> Run code in sandbox if response includes code
 9. Detect    -> Scan for anti-patterns if user provided code
10. Score     -> Compute composite confidence
11. Update    -> Save session memory, mark pipeline progress, update skill level
12. Suggest   -> Generate follow-up question suggestions
```

---

## Project Structure

```
ds_mentor/
+-- core/                          # Core AI engine (11 modules)
|   +-- agent.py                   #   ReAct orchestrator
|   +-- retriever.py               #   Hybrid BM25 + Dense + RRF + Cross-encoder
|   +-- code_engine.py             #   AST-validated sandboxed execution
|   +-- generator.py               #   Template + optional LLM response generation
|   +-- stage_classifier.py        #   Keyword-weighted pipeline classifier
|   +-- pipeline_tracker.py        #   7-stage tracking with next-step suggestions
|   +-- confidence_scorer.py       #   Composite reliability scoring
|   +-- antipattern_detector.py    #   Code anti-pattern detection (7 patterns)
|   +-- memory.py                  #   Session memory + user profiles
|   +-- skill_assessor.py          #   Adaptive difficulty
|   +-- dataset_profiler.py        #   CSV/Excel schema detection and summary
|   +-- why_engine.py              #   WHY/WHEN explanation extraction
|   +-- question_generator.py      #   Follow-up question suggestions
|   +-- socratic_engine.py         #   Guided discovery mode
|   +-- __init__.py
|
+-- data/                          # Knowledge base
|   +-- generate_dataset_simple.py #   Dataset generator (pure stdlib)
|   +-- dataset.csv                #   109+ QA pairs across 7 stages [generated]
|
+-- services/                      # API layer
|   +-- api.py                     #   FastAPI application
|   +-- routes/
|       +-- chat.py                #   POST /chat
|       +-- upload.py              #   POST /upload (with dataset profiling)
|       +-- feedback.py            #   POST /feedback
|       +-- health.py              #   GET /health
|
+-- ui/                            # Frontend
|   +-- streamlit_app.py           #   Streamlit chat interface
|
+-- tests/                         # Test suite (52 tests)
|   +-- test_agent.py
|   +-- test_retriever.py
|   +-- test_code_engine.py
|   +-- test_stage_classifier.py
|   +-- test_pipeline_tracker.py
|   +-- test_dataset_profiler.py
|   +-- test_why_engine.py
|   +-- test_question_generator.py
|   +-- test_socratic_engine.py
|   +-- smoke_test.py              #   End-to-end integration (11 subsystems)
|
+-- models/configs/                # YAML configurations
|   +-- retriever_config.yaml
|   +-- generator_config.yaml
|
+-- storage/                       # Persistence layer
|   +-- session_store.py
|   +-- vector_store.py
|
+-- eval/                          # Evaluation framework
|   +-- run_eval.py
|
+-- requirements.txt
+-- README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the Knowledge Base

```bash
python data/generate_dataset_simple.py
```

Creates `data/dataset.csv` with 109+ curated QA pairs across all 7 pipeline stages. Uses only Python stdlib.

### 3. Launch the UI

```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501`. The app runs with BM25-only retrieval out of the box (zero model downloads).

### 4. Run Tests

```bash
# Full test suite (52 tests)
python -m pytest tests/ -v

# End-to-end smoke test (11 subsystems)
python tests/smoke_test.py
```

---

## Test Results

### Unit and Integration Tests: 52/52 pass

| Test File | Tests | Status |
|-----------|-------|--------|
| test_pipeline_tracker.py | 2 | Pass |
| test_stage_classifier.py | 2 | Pass |
| test_code_engine.py | 2 | Pass |
| test_retriever.py | 2 | Pass |
| test_agent.py | 2 | Pass |
| test_dataset_profiler.py | 9 | Pass |
| test_why_engine.py | 7 | Pass |
| test_question_generator.py | 6 | Pass |
| test_socratic_engine.py | 9 | Pass |
| smoke_test.py | 11 | Pass |

---

## Dataset Upload

The system supports dataset-aware responses:

1. Upload a CSV or Excel file via the sidebar in the Streamlit UI (or POST to `/upload`).
2. The `DatasetProfiler` automatically detects:
   - Column names and data types
   - Missing value counts and percentages per column
   - Numeric vs categorical column separation
   - Likely target column (heuristic)
   - Top values for categorical columns, mean/std/min/max for numeric
3. This profile is injected into the response generator so answers reference your actual data.

---

## The 7-Stage Data Science Pipeline

| Stage | Name | What It Covers |
|-------|------|----------------|
| 1 | Problem Understanding | Define objective, target, metrics, baseline |
| 2 | Data Loading | Read CSV/JSON/SQL, check shape, dtypes |
| 3 | Exploratory Data Analysis | Distributions, correlations, outliers, missing patterns |
| 4 | Preprocessing | Imputation, outlier capping, scaling, encoding |
| 5 | Feature Engineering | Polynomial features, PCA, datetime extraction |
| 6 | Modeling | Random Forest, XGBoost, cross-validation, tuning |
| 7 | Evaluation | AUC-ROC, confusion matrix, learning curves |

Pipeline Guardian warns when users skip stages and suggests the logical next step.

---

## Configuration

### Default (zero-setup)

```python
{
    "retriever": {
        "use_dense": False,
        "use_cross_encoder": False,
        "stage_boost": 1.3,
    },
    "generator": {"provider": "template"},
}
```

### Full mode (with model downloads)

```python
{
    "retriever": {
        "use_dense": True,
        "embed_model": "all-MiniLM-L6-v2",
        "use_cross_encoder": True,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "generator": {"provider": "openai", "model": "gpt-3.5-turbo"},
}
```

### Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| OPENAI_API_KEY | No | -- | OpenAI LLM for response generation |
| LLM_PROVIDER | No | template | template or openai |
| LLM_MODEL | No | gpt-3.5-turbo | OpenAI model name |

---

## Code Safety

The sandbox enforces:

- **Import whitelist**: pandas, numpy, sklearn, matplotlib, seaborn, scipy, math, statistics, collections, itertools, json, csv, warnings, os, re, datetime, time, typing, functools, operator
- **Blocked operations**: subprocess, os.system, eval(), exec(), \_\_import\_\_, requests, socket, pickle.loads
- **Timeout**: 30-second execution limit
- **Isolation**: Runs in a subprocess with restricted builtins
