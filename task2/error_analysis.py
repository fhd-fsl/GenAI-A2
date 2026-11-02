"""Task 2: Error Analysis for Criterion-Level Eligibility.

Categorizes and presents representative errors, contradictory cases,
and failures due to missing information. Per assignment: analyze
at least eight distinct errors.
"""

from typing import Any
from collections import defaultdict
import json


def categorize_errors(predictions: list[dict]) -> dict[str, list[dict]]:
    """
    Categorize prediction errors into distinct types.

    Returns dict mapping error category to list of example predictions.
    """
    categories = defaultdict(list)

    for p in predictions:
        gold = p.get("gold_label", "")
        pred = p.get("prediction", "")
        gold_ev = set(p.get("gold_evidence", []))
        pred_ev = set(p.get("evidence_sentence_ids", []))

        if gold == pred:
            continue  # Correct prediction

        # Category 1: Wrong label, correct evidence
        if gold_ev and pred_ev and gold_ev == pred_ev:
            categories["wrong_label_correct_evidence"].append(p)
            continue

        # Category 2: Correct label, wrong evidence
        if gold_ev and pred_ev and gold_ev != pred_ev:
            categories["correct_label_wrong_evidence"].append(p)
            continue

        # Category 3: Missing evidence (gold has evidence, pred empty)
        if gold_ev and not pred_ev:
            categories["missing_evidence"].append(p)
            continue

        # Category 4: Hallucinated evidence (pred has evidence, gold empty)
        if not gold_ev and pred_ev:
            categories["hallucinated_evidence"].append(p)
            continue

        # Category 5: Confusion between included / not enough information
        if {gold, pred} <= {"included", "not enough information"}:
            categories["included_vs_not_enough_info"].append(p)
            continue

        # Category 6: Confusion between not included / not applicable
        if {gold, pred} <= {"not included", "not applicable"}:
            categories["not_included_vs_not_applicable"].append(p)
            continue

        # Category 7: Parse errors
        if pred == "parse_error":
            categories["parse_error"].append(p)
            continue

        # Category 8: Empty evidence for definitive prediction
        if pred in {"included", "excluded", "not included", "not excluded"} and not pred_ev:
            categories["empty_evidence_definitive_prediction"].append(p)
            continue

        # Category 9: Exclusion criteria predicted as inclusion label (or vice versa)
        if (gold.startswith("not ") != pred.startswith("not ")) and gold != pred:
            categories["polarity_confusion"].append(p)
            continue

        # Default: other
        categories["other"].append(p)

    return dict(categories)


def select_representative_errors(categories: dict[str, list[dict]], max_per_cat: int = 2) -> list[dict]:
    """Select up to max_per_cat examples from each category for report."""
    representatives = []
    for cat, examples in categories.items():
        if not examples:
            continue
        representatives.append({
            "category": cat,
            "count": len(examples),
            "examples": examples[:max_per_cat],
        })
    return representatives


def format_error_for_report(example: dict) -> str:
    """Format a single error example for report inclusion."""
    return f"""
**Criterion:** {example.get('criterion_id', 'N/A')}
**Type:** {example.get('criterion_type', 'N/A')}
**Criterion Text:** {example.get('criterion_text', 'N/A')[:200]}...
**Gold Label:** {example.get('gold_label', 'N/A')}
**Predicted:** {example.get('prediction', 'N/A')}
**Gold Evidence IDs:** {example.get('gold_evidence', [])}
**Pred Evidence IDs:** {example.get('evidence_sentence_ids', [])}
**Gold Evidence Text:** {example.get('gold_evidence_text', 'N/A')[:200] if 'gold_evidence_text' in example else 'N/A'}
**Pred Evidence Text:** {example.get('evidence_text', 'N/A')[:200]}
"""


def analyze_contradictory_cases(predictions: list[dict]) -> list[dict]:
    """
    Find contradictory cases: same patient, similar criteria, different outcomes
    that may indicate inconsistency.
    """
    by_patient = defaultdict(list)
    for p in predictions:
        by_patient[p.get("patient_id", "")].append(p)

    contradictions = []
    for patient_id, preds in by_patient.items():
        # Look for same criterion text with different predictions
        by_criterion = defaultdict(list)
        for p in preds:
            by_criterion[p.get("criterion_text", "")].append(p)

        for criterion_text, criterion_preds in by_criterion.items():
            if len(criterion_preds) > 1:
                preds_set = set(p.get("prediction") for p in criterion_preds)
                if len(preds_set) > 1:
                    contradictions.append({
                        "patient_id": patient_id,
                        "criterion_text": criterion_text,
                        "predictions": criterion_preds,
                    })
    return contradictions


def analyze_missing_information_failures(predictions: list[dict]) -> list[dict]:
    """
    Find failures due to missing information: gold label is 'not enough information'
    but model predicts definitive label, or vice versa.
    """
    failures = []
    for p in predictions:
        gold = p.get("gold_label", "")
        pred = p.get("prediction", "")

        # Case 1: Gold = not enough info, but model makes definitive prediction
        if gold == "not enough information" and pred not in {"not enough information", "parse_error"}:
            failures.append({
                "type": "definitive_vs_not_enough_info",
                "example": p,
            })

        # Case 2: Gold = definitive, but model says not enough info
        if gold != "not enough information" and pred == "not enough information":
            failures.append({
                "type": "not_enough_info_vs_definitive",
                "example": p,
            })
    return failures


def generate_error_analysis_report(predictions: list[dict], output_path: str) -> None:
    """Generate full error analysis report as markdown."""
    categories = categorize_errors(predictions)
    representatives = select_representative_errors(categories)
    contradictions = analyze_contradictory_cases(predictions)
    missing_info = analyze_missing_information_failures(predictions)

    with open(output_path, "w") as f:
        f.write("# Task 2 Error Analysis Report\n\n")
        f.write(f"Total predictions: {len(predictions)}\n")
        f.write(f"Total errors: {sum(len(v) for v in categories.values())}\n\n")

        f.write("## Error Categories\n\n")
        for rep in representatives:
            f.write(f"### {rep['category'].replace('_', ' ').title()} ({rep['count']} cases)\n\n")
            for ex in rep["examples"]:
                f.write(format_error_for_report(ex))
                f.write("\n---\n\n")

        f.write("## Contradictory Cases\n\n")
        f.write(f"Found {len(contradictions)} contradictory cases.\n\n")
        for c in contradictions[:5]:
            f.write(f"- **Patient {c['patient_id']}**: \"{c['criterion_text'][:100]}...\" -> ")
            f.write(f"Predictions: {[p.get('prediction') for p in c['predictions']]}\n")
        f.write("\n")

        f.write("## Missing Information Failures\n\n")
        f.write(f"Total: {len(missing_info)} cases\n\n")
        by_type = defaultdict(list)
        for m in missing_info:
            by_type[m["type"]].append(m["example"])
        for mtype, examples in by_type.items():
            f.write(f"### {mtype.replace('_', ' ').title()} ({len(examples)} cases)\n\n")
            for ex in examples[:3]:
                f.write(format_error_for_report(ex))
                f.write("\n---\n\n")

        f.write("## Summary: 8+ Distinct Error Types\n\n")
        for i, (cat, examples) in enumerate(categories.items(), 1):
            if examples:
                f.write(f"{i}. **{cat.replace('_', ' ').title()}**: {len(examples)} cases\n")

    print(f"Error analysis saved to {output_path}")


if __name__ == "__main__":
    # Test with sample predictions
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from task1.eval.metrics import compute_all_metrics

    # Load actual test predictions
    pred_path = "task1/outputs/eval/test_predictions.json"
    with open(pred_path) as fp:
        preds = json.load(fp)

    generate_error_analysis_report(preds, "task2/outputs/error_analysis.md")