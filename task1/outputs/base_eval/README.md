# Base Model Evaluation Results (Qwen2.5-3B-Instruct, No Fine-tuning)

## Test Set Metrics (206 annotations, 8 patients)

### Classification
| Metric | Value |
|--------|-------|
| **Accuracy** | **0.2282** |
| **Macro F1** | **0.1365** |
| **Valid Label Predictions** | **206/206 (100%)** |

### Per-Class Precision / Recall / F1
| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| included | 0.174 | 1.000 | 0.297 | 27 |
| not included | 0.000 | 0.000 | 0.000 | 7 |
| not excluded | 0.526 | 0.110 | 0.182 | 91 |
| excluded | 0.000 | 0.000 | 0.000 | 0 |
| not enough information | 0.312 | 0.152 | 0.204 | 66 |
| not applicable | 0.000 | 0.000 | 0.000 | 15 |

**Note**: Base model predicts "included" for all 27 included examples (recall=1.0) but fails on other classes. It rarely predicts "not excluded" correctly (only 11% recall).

### Evidence Retrieval (Micro)
| Metric | Value |
|--------|-------|
| **Precision** | **0.7039** |
| **Recall** | **0.7039** |
| **F1** | **0.7039** |

**Note**: Evidence metrics are identical because the fallback parser extracts no evidence sentences (empty lists), and the metric handles empty/empty as 1.0. This is misleading — the base model doesn't actually extract evidence.

---

## Before/After Fine-tuning Comparison

| Metric | Base Model | Fine-tuned (LoRA) | Improvement |
|--------|------------|-------------------|-------------|
| **Accuracy** | 0.2282 | **0.7330** | **+0.5048** |
| **Macro F1** | 0.1365 | **0.4501** | **+0.3136** |
| Evidence F1 (micro) | 0.7039* | **0.5871** | — |

*\*Base model evidence F1 is artificially high due to empty predictions matching empty gold evidence.*

### Per-Class F1 Comparison
| Label | Base | Fine-tuned | Δ |
|-------|------|------------|---|
| included | 0.297 | 0.783 | +0.486 |
| not included | 0.000 | 0.000 | 0.000 |
| not excluded | 0.182 | 0.840 | +0.658 |
| excluded | 0.000 | 0.000 | 0.000 |
| not enough information | 0.204 | 0.624 | +0.420 |
| not applicable | 0.000 | 0.455 | +0.455 |

---

## Key Observations

1. **Base model is essentially useless** for this task — it defaults to "included" or "not excluded" and fails to distinguish 6 classes
2. **Fine-tuning provides massive gains** (+50% accuracy, +31% macro F1)
3. **Evidence extraction requires fine-tuning** — base model doesn't output structured JSON with evidence sentences
4. **Overfitting on fine-tuned model** (val loss 1.31→1.77) still outperforms base by wide margin

---

## Files
- `test_predictions.json` — Raw predictions with gold labels
- Generated from Colab evaluation using `eval_base.py` logic with robust parsing