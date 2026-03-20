# DS Mentor QA System — Evaluation Results

## Comparison vs Baselines

| Metric | Base CodeT5 | ChatGPT-3.5 | **Our System** | vs ChatGPT |
|--------|------------|-------------|---------------|------------|
| Stage Clf Accuracy | 41.0% | 68.0% | **76.9%** | +13.1% |
| Stage Clf Macro-F1 | 38.0% | 65.0% | **75.1%** | +15.6% |
| Retrieval MRR | 43.0% | 71.0% | **96.9%** | +36.5% |
| Retrieval Recall@5 | 55.0% | 78.0% | **100.0%** | +28.2% |
| Retrieval Latency (ms) | 12.00 | 850.00 | **6.32** | +99.3% faster |
| CodeBLEU Score | 31.0% | 52.0% | **69.0%** | +32.7% |
| Code Syntax Rate | 61.0% | 79.0% | **98.3%** | +24.4% |
| Visualization Rate | 0.0% | 31.0% | **25.0%** | -19.4% |
| Skip Detection Acc | 0.0% | 0.0% | **100.0%** | — |
| Conv Accuracy | 20.0% | 65.0% | **100.0%** | +53.8% |
| Intent Accuracy | 0.0% | 60.0% | **100.0%** | +66.7% |

## Key Novelty Claims (vs ChatGPT)

| Claim | Evidence |
|-------|---------|
| Grounded answers (no hallucination) | BM25+TF-IDF retrieval from curated 701-example KB |
| Domain-specific stage awareness | Stage classifier trained on DS-specific data |
| Pipeline workflow tracking | 100% skip detection accuracy |
| Multi-turn context resolution | Pronoun/reference resolution across turns |
| Live code adaptation | Context extractor adapts code to column names/models |
| Sub-10ms retrieval | vs 850ms for ChatGPT API calls |
