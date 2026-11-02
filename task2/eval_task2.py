"""Task 2: Evaluation Runner.

Runs the eligibility pipeline on test/validation splits,
computes all required metrics, and generates error analysis.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from task2.pipeline import run_evaluation
from task2.metrics import compute_all_metrics, print_metrics
from task2.error_analysis import generate_error_analysis_report


def main():
    parser = argparse.ArgumentParser(description="Task 2: Criterion-Level Eligibility Evaluation")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--output-dir", default="task2/outputs", help="Output directory")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    print(f"Running Task 2 evaluation on {args.split} split...")
    predictions = run_evaluation(
        adapter_path=str(adapter_path),
        split=args.split,
        output_dir=output_dir,
    )

    # Compute metrics
    print("\n--- Computing Metrics ---")
    metrics = compute_all_metrics(predictions)
    print_metrics(metrics)

    # Save metrics
    metrics_path = output_dir / f"{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate error analysis
    print("\n--- Generating Error Analysis ---")
    error_path = output_dir / f"{args.split}_error_analysis.md"
    generate_error_analysis_report(predictions, str(error_path))

    print(f"\n✅ Task 2 evaluation complete!")
    print(f"  Predictions: {output_dir}/{args.split}_predictions.json")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Error Analysis: {error_path}")


if __name__ == "__main__":
    main()