# Task 1 Evaluation Results (Fine-tuned Model)

## Test Set Metrics (206 annotations, 8 patients)

### Classification
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.7330** |
| **Macro F1** | **0.4501** |

### Per-Class Precision / Recall / F1
| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| included | 0.947 | 0.667 | 0.783 | 27 |
| not included | 0.000 | 0.000 | 0.000 | 7 |
| not excluded | 0.736 | 0.978 | 0.840 | 91 |
| excluded | 0.000 | 0.000 | 0.000 | 0 |
| not enough information | 0.661 | 0.591 | 0.624 | 66 |
| not applicable | 0.714 | 0.333 | 0.455 | 15 |

### Confusion Matrix
Rows = gold, Cols = pred | incl | not-incl | not-excl | excl | not-enough | not-appl
|---|---|---|---|---|---|---|
| **included** | 18 | 0 | 0 | 0 | 9 | 0 |
| **not included** | 0 | 0 | 0 | 0 | 7 | 0 |
| **not excluded** | 0 | 0 | 89 | 0 | 1 | 1 |
| **excluded** | 0 | 0 | 0 | 0 | 0 | 0 |
| **not enough info** | 1 | 0 | 25 | 0 | 39 | 1 |
| **not applicable** | 0 | 0 | 7 | 0 | 3 | 5 |

### Evidence Retrieval (Micro)
| Metric | Value |
|--------|-------|
| **Precision** | **0.5514** |
| **Recall** | **0.6277** |
| **F1** | **0.5871** |

### Evidence F1 per Class
| Label | F1 |
|-------|-----|
| included | 0.831 |
| not included | 0.400 |
| not excluded | 0.596 |
| excluded | 0.000 |
| not enough information | 0.313 |
| not applicable | 0.500 |

### Inclusion vs Exclusion Breakdown
| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Inclusion | 0.7333 | 0.2616 |
| Exclusion | 0.7328 | 0.2492 |

---

**Generated from**: `task1/colab/submission-package/adapter/test_metrics_detailed.json`