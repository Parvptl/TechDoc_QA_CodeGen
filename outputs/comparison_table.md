# DS Mentor QA System — Evaluation Results

## Comparison vs Baselines

| Metric | Base CodeT5 | ChatGPT-3.5 | **Our System** | vs ChatGPT |
|--------|------------|-------------|---------------|------------|
| Stage Clf Accuracy | 41.0% | 68.0% | **73.1%** | +7.5% |
| Stage Clf Macro-F1 | 38.0% | 65.0% | **64.8%** | -0.4% |
| Stage Clf Weighted-F1 | — | — | **67.2%** | — |
| Stage Clf Balanced Acc | — | — | **70.2%** | — |
| Intent Macro-F1 | — | — | **100.0%** | — |
| Intent Macro-Precision | — | — | **100.0%** | — |
| Intent Macro-Recall | — | — | **100.0%** | — |
| Retrieval MRR | 43.0% | 71.0% | **79.2%** | +11.5% |
| Retrieval Recall@1 | — | — | **65.4%** | — |
| Retrieval Recall@3 | — | — | **92.3%** | — |
| Retrieval Recall@5 | 55.0% | 78.0% | **96.2%** | +23.3% |
| Retrieval Latency (ms) | 12.00 | 850.00 | **2.36** | +99.7% faster |
| Retrieval Top1 Stage Acc | — | — | **65.4%** | — |
| Retrieval Avg Top1 Score | — | — | **0.03** | — |
| Retrieval Avg Top1-Top2 Margin | — | — | **0.00** | — |
| CodeBLEU Score | 31.0% | 52.0% | **69.2%** | +33.2% |
| CodeBLEU StdDev | — | — | **0.13** | — |
| Code Syntax Rate | 61.0% | 79.0% | **100.0%** | +26.6% |
| Generated Code Syntax Rate | — | — | **100.0%** | — |
| Code Keyword Hit Rate | — | — | **60.0%** | — |
| Avg Generated LOC | — | — | **16.60** | — |
| Visualization Rate | 0.0% | 31.0% | **75.0%** | +141.9% |
| Visualization Avg Latency (ms) | — | — | **4025.50** | — |
| Visualization P95 Latency (ms) | — | — | **5574.75** | — |
| Visualization Error Rate | — | — | **0.25** | — |
| Skip Detection Acc | 0.0% | 0.0% | **100.0%** | — |
| Conv Accuracy | 20.0% | 65.0% | **100.0%** | +53.8% |
| Conv Resolution Accuracy | — | — | **100.0%** | — |
| Conv Joint Accuracy | — | — | **100.0%** | — |
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
