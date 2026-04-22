## DS Mentor Pro — Evaluation Suite

### Retrieval

| Metric | Value |
|---|---:|
| Precision@1 | 0.5000 |
| Precision@3 | 0.1667 |
| Precision@5 | 0.1000 |
| Recall@1 | 0.5000 |
| Recall@3 | 0.5000 |
| Recall@5 | 0.5000 |
| MRR | 0.5000 |
| nDCG@5 | 0.5000 |

### Stage classification

| Metric | Value |
|---|---:|
| Accuracy | 0.6000 |
| Macro-F1 | 0.5667 |

**Confusion matrix (stages 1–7)**

| true\pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 2 | 1 | 0 | 0 |
| 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |

### Anti-pattern detection

| Metric | Value |
|---|---:|
| Precision (micro) | 1.0000 |
| Recall (micro) | 0.8000 |
| F1 (micro) | 0.8889 |

Per-pattern PRF1:

| Pattern id | Precision | Recall | F1 |
|---|---:|---:|---:|
| `accuracy_imbalanced_warning` | 0.0000 | 0.0000 | 0.0000 |
| `blind_dropna` | 1.0000 | 1.0000 | 1.0000 |
| `data_leakage_fit_transform_before_split` | 0.0000 | 0.0000 | 0.0000 |
| `feature_selection_before_split` | 1.0000 | 1.0000 | 1.0000 |
| `fit_on_test_set` | 1.0000 | 1.0000 | 1.0000 |
| `no_validation_strategy` | 0.0000 | 0.0000 | 0.0000 |
| `predict_on_train_only` | 1.0000 | 1.0000 | 1.0000 |

### Response quality

| Metric | Value |
|---|---:|
| BLEU-1 | 0.4329 |
| ROUGE-L | 0.4738 |

