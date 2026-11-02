"""Task 2: Extended Metrics for Criterion-Level Eligibility.

Includes overall metrics, inclusion vs exclusion separate analysis,
and evidence retrieval metrics with proper empty-set handling.
"""

from collections import defaultdict
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# All 6 eligibility labels (preserved per assignment)
ELIGIBILITY_LABELS = [
    "included",
    "not included",
    "not excluded",
    "excluded",
    "not enough information",
    "not applicable",
]

LABEL_TO_IDX = {label: i for i, label in enumerate(ELIGIBILITY_LABELS)}


def evidence_f1(gold: list[int], pred: list[int]) -> tuple[float, float, float]:
    """
    Compute precision, recall, F1 for evidence sentence retrieval.

    Handles empty gold/predicted sets per assignment requirements:
    - Empty gold + empty pred = 1.0 (correctly identified no evidence needed)
    - Empty gold + non-empty pred = 0.0 (hallucinated evidence)
    - Non-empty gold + empty pred = 0.0 (missed evidence)
    """
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0

    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> dict[str, Any]:
    """Compute classification metrics: accuracy, macro-F1, per-class P/R/F1, confusion matrix."""
    # Filter valid predictions (both gold and pred in label set)
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t >= 0 and p >= 0]
    if not valid_pairs:
        return {"error": "No valid predictions"}

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    metrics = {
        "n_total": len(y_true),
        "n_valid": len(valid_pairs),
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "macro_f1": float(f1_score(y_true_valid, y_pred_valid, average="macro")),
    }

    # Per-class metrics
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average=None, labels=range(len(ELIGIBILITY_LABELS)), zero_division=0
    )

    per_class = {}
    for i, label in enumerate(ELIGIBILITY_LABELS):
        per_class[label] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }
    metrics["per_class"] = per_class

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(ELIGIBILITY_LABELS)))
    metrics["confusion_matrix"] = {
        "labels": ELIGIBILITY_LABELS,
        "matrix": cm.tolist(),
    }

    return metrics


def compute_evidence_metrics(predictions: list[dict]) -> dict[str, Any]:
    """
    Compute evidence retrieval metrics (micro-averaged and per-class).

    Returns micro P/R/F1 and per-class evidence F1.
    """
    all_prec = []
    all_rec = []
    all_f1 = []
    class_f1 = defaultdict(list)

    for p in predictions:
        gold_ev = p.get("gold_evidence", [])
        pred_ev = p.get("evidence_sentence_ids", [])
        prec, rec, f1 = evidence_f1(gold_ev, pred_ev)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        class_f1[p.get("gold_label", "unknown")].append(f1)

    metrics = {
        "evidence_precision_micro": float(np.mean(all_prec)) if all_prec else 0.0,
        "evidence_recall_micro": float(np.mean(all_rec)) if all_rec else 0.0,
        "evidence_f1_micro": float(np.mean(all_f1)) if all_f1 else 0.0,
        "per_class_evidence_f1": {
            label: float(np.mean(f1s)) if f1s else 0.0
            for label, f1s in class_f1.items()
        },
    }
    return metrics


def compute_inclusion_exclusion_metrics(predictions: list[dict]) -> dict[str, Any]:
    """
    Compute metrics separately for inclusion and exclusion criteria.

    Per assignment: "reported both overall and separately for inclusion versus exclusion criteria"
    """
    incl_preds = [p for p in predictions if p.get("criterion_type") == "inclusion"]
    excl_preds = [p for p in predictions if p.get("criterion_type") == "exclusion"]

    def compute_split(preds: list[dict]) -> dict[str, Any]:
        if not preds:
            return {"n_samples": 0}
        y_true = [LABEL_TO_IDX.get(p.get("gold_label", ""), -1) for p in preds]
        y_pred = [LABEL_TO_IDX.get(p.get("prediction", ""), -1) for p in preds]
        cls = compute_classification_metrics(y_true, y_pred)
        ev = compute_evidence_metrics(preds)
        return {**cls, **ev, "n_samples": len(preds)}

    return {
        "inclusion": compute_split(incl_preds),
        "exclusion": compute_split(excl_preds),
    }


def compute_all_metrics(predictions: list[dict]) -> dict[str, Any]:
    """Compute all metrics from predictions list (as saved by evaluation script)."""
    y_true = []
    y_pred = []

    for p in predictions:
        y_true.append(LABEL_TO_IDX.get(p.get("gold_label", ""), -1))
        y_pred.append(LABEL_TO_IDX.get(p.get("prediction", ""), -1))

    cls_metrics = compute_classification_metrics(y_true, y_pred)
    ev_metrics = compute_evidence_metrics(predictions)
    ie_metrics = compute_inclusion_exclusion_metrics(predictions)

    return {**cls_metrics, **ev_metrics, "inclusion_exclusion": ie_metrics}


def print_metrics(metrics: dict[str, Any]) -> None:
    """Pretty-print metrics to console."""
    print(f"Valid predictions: {metrics.get('n_valid', 'N/A')}/{metrics.get('n_total', 'N/A')}")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1: {metrics.get('macro_f1', 0):.4f}")
    print()

    print("Per-class metrics:")
    for label, vals in metrics.get("per_class", {}).items():
        print(f"  {label}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f} (n={vals['support']})")
    print()

    print("Confusion Matrix:")
    cm = metrics.get("confusion_matrix", {})
    if cm:
        labels = cm.get("labels", [])
        matrix = cm.get("matrix", [])
        print("      " + "  ".join(l[:4] for l in labels))
        for i, row in enumerate(matrix):
            print(f"{labels[i][:4]}: " + "  ".join(f"{x:3d}" for x in row))
    print()

    print("Evidence Retrieval (micro):")
    print(f"  Precision: {metrics.get('evidence_precision_micro', 0):.4f}")
    print(f"  Recall:    {metrics.get('evidence_recall_micro', 0):.4f}")
    print(f"  F1:        {metrics.get('evidence_f1_micro', 0):.4f}")
    print()

    print("Per-class Evidence F1:")
    for label, f1 in metrics.get("per_class_evidence_f1", {}).items():
        print(f"  {label}: F1={f1:.4f}")
    print()

    # Inclusion vs Exclusion
    ie = metrics.get("inclusion_exclusion", {})
    if ie:
        print("=== Inclusion vs Exclusion ===")
        for split in ["inclusion", "exclusion"]:
            split_metrics = ie.get(split, {})
            print(f"\n{split.capitalize()} (n={split_metrics.get('n_samples', 0)}):")
            if split_metrics.get('n_samples', 0) > 0:
                print(f"  Accuracy: {split_metrics.get('accuracy', 0):.4f}")
                print(f"  Macro F1: {split_metrics.get('macro_f1', 0):.4f}")
                print(f"  Evidence F1: {split_metrics.get('evidence_f1_micro', 0):.4f}")