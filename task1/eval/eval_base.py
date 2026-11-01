"""Base Model Evaluation for TrialGPT Criterion-Level Eligibility.

Evaluates Qwen2.5-3B-Instruct WITHOUT LoRA adapter on val/test split.
Used for before/after fine-tuning comparison (Rubric B4).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.dataset_interface import TrialGPTDataset, SplitName
from task1.eval.metrics import compute_all_metrics, print_metrics, ELIGIBILITY_LABELS
from task1.preprocessing.tokenizer import (
    SYSTEM_PROMPT,
    parse_expert_sentences,
    split_note_into_sentences,
    build_criterion_id,
    MAX_SEQ_LEN,
)


def generate_prediction(
    model,
    tokenizer,
    note: str,
    criterion_type: str,
    criterion_text: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate a single prediction using greedy decoding."""
    user_prompt = (
        f"Patient Description:\n{note}\n\n"
        f"Criterion Type: {criterion_type}\n"
        f"Criterion Text: {criterion_text}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated


def parse_prediction_json(pred_text: str) -> dict:
    """Parse model output as JSON, with fallback for parse errors."""
    try:
        return json.loads(pred_text)
    except json.JSONDecodeError:
        return {
            "prediction": "parse_error",
            "evidence_sentence_ids": [],
            "evidence_text": [],
            "explanation": pred_text,
        }


def evaluate_split(
    model,
    tokenizer,
    split_name: SplitName,
    output_dir: Path,
) -> list[dict]:
    """Evaluate model on a dataset split and return predictions with gold labels."""
    dataset = TrialGPTDataset()
    split_ds = dataset.get_split(split_name)

    predictions = []

    for ex in tqdm(split_ds, desc=f"Evaluating base model on {split_name}"):
        pred_text = generate_prediction(model, tokenizer, ex["note"], ex["criterion_type"], ex["criterion_text"])
        pred_json = parse_prediction_json(pred_text)

        # Gold evidence
        gold_evidence = parse_expert_sentences(ex["expert_sentences"])

        predictions.append({
            "annotation_id": ex["annotation_id"],
            "patient_id": ex["patient_id"],
            "criterion_id": build_criterion_id(ex["trial_id"], ex["criterion_type"], ex["annotation_id"]),
            "criterion_type": ex["criterion_type"],
            "criterion_text": ex["criterion_text"],
            "prediction": pred_json.get("prediction", ""),
            "evidence_sentence_ids": pred_json.get("evidence_sentence_ids", []),
            "evidence_text": pred_json.get("evidence_text", []),
            "explanation": pred_json.get("explanation", ""),
            "gold_label": ex["expert_eligibility"],
            "gold_evidence": gold_evidence,
        })

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{split_name}_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {pred_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate BASE Qwen2.5-3B-Instruct (no adapter)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="Split to evaluate")
    parser.add_argument("--output-dir", default="results/task1/base_eval", help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for generation")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Evaluate
    predictions = evaluate_split(model, tokenizer, args.split, Path(args.output_dir))

    # Compute metrics
    print("\n--- Computing Metrics (Base Model) ---")
    metrics = compute_all_metrics(predictions)
    print_metrics(metrics)

    # Save metrics
    metrics_path = Path(args.output_dir) / f"{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()