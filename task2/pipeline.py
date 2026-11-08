"""Task 2: Criterion-Level Eligibility and Evidence Verification Pipeline.

Reusable inference pipeline for fine-tuned TrialGPT model.
Returns structured predictions with criterion, category, evidence, and explanation.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_interface import TrialGPTDataset
from task1.preprocessing.tokenizer import (
    SYSTEM_PROMPT,
    parse_expert_sentences,
    split_note_into_sentences,
    build_criterion_id,
    MAX_SEQ_LEN,
)


class EligibilityPipeline:
    """Pipeline for criterion-level eligibility prediction with evidence."""

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
    ):
        """
        Initialize pipeline with fine-tuned adapter.

        Args:
            adapter_path: Path to LoRA adapter directory
            base_model: Base model name or path
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model on {self.device}...")
        self._load_model()

    def _load_model(self):
        """Load base model + LoRA adapter."""
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        weights_file_exists = (
            self.adapter_path.is_dir()
            and (
                (self.adapter_path / "adapter_model.safetensors").exists()
                or (self.adapter_path / "adapter_model.bin").exists()
            )
        )

        if weights_file_exists:
            self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path), use_fast=True)
        else:
            print(
                f"[INFO] LoRA weights not found in '{self.adapter_path}'.\n"
                f"       Falling back to base model '{self.base_model}'.\n"
                f"       (Download fine-tuned adapter weights from Google Drive for full clinical accuracy)."
            )
            self.model = base
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)

        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Ensure chat template is set (Qwen2.5 chat template)
        if self.tokenizer.chat_template is None:
            # Load from base model tokenizer which has the template
            base_tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
            self.tokenizer.chat_template = base_tokenizer.chat_template

    def predict(
        self,
        note: str,
        criterion_type: str,
        criterion_text: str,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Predict eligibility for a single criterion.

        Returns:
            {
                "criterion_id": str,
                "criterion_type": str,
                "criterion_text": str,
                "prediction": str,  # one of 6 eligibility labels
                "evidence_sentence_ids": List[int],
                "evidence_text": List[str],
                "explanation": str,
            }
        """
        # Build prompt
        user_prompt = (
            f"Patient Description:\n{note}\n\n"
            f"Criterion Type: {criterion_type}\n"
            f"Criterion Text: {criterion_text}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        # Parse output
        pred_json = self._parse_prediction(generated)

        # Extract evidence text from note
        sentences = split_note_into_sentences(note)
        evidence_ids = pred_json.get("evidence_sentence_ids", [])
        evidence_text = [
            sentences[i] for i in evidence_ids if i < len(sentences)
        ]

        return {
            "criterion_id": pred_json.get("criterion_id", ""),
            "criterion_type": criterion_type,
            "criterion_text": criterion_text,
            "prediction": pred_json.get("prediction", "parse_error"),
            "evidence_sentence_ids": evidence_ids,
            "evidence_text": evidence_text,
            "explanation": pred_json.get("explanation", ""),
        }

    def _parse_prediction(self, text: str) -> dict:
        """Parse model output as JSON with fallback."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract prediction label from text
            for label in [
                "included", "not included", "not excluded", "excluded",
                "not enough information", "not applicable"
            ]:
                if label in text.lower():
                    return {"prediction": label, "evidence_sentence_ids": [], "explanation": text}
            return {
                "prediction": "parse_error",
                "evidence_sentence_ids": [],
                "explanation": text,
            }

    def predict_batch(
        self,
        items: list[dict],
        max_new_tokens: int = 512,
    ) -> list[dict]:
        """
        Predict for multiple criteria.

        Args:
            items: List of {"note": str, "criterion_type": str, "criterion_text": str, ...}
        """
        results = []
        for item in items:
            pred = self.predict(
                item["note"],
                item["criterion_type"],
                item["criterion_text"],
                max_new_tokens,
            )
            # Include original metadata
            pred.update({
                "annotation_id": item.get("annotation_id"),
                "patient_id": item.get("patient_id"),
                "trial_id": item.get("trial_id"),
            })
            results.append(pred)
        return results


def run_evaluation(
    adapter_path: str,
    split: str = "test",
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Run full evaluation on a dataset split.

    Returns list of predictions with gold labels for metrics computation.
    """
    pipeline = EligibilityPipeline(adapter_path)
    dataset = TrialGPTDataset()
    split_ds = dataset.get_split(split)

    predictions = []
    for ex in split_ds:
        pred = pipeline.predict(ex["note"], ex["criterion_type"], ex["criterion_text"])

        # Add gold labels
        pred["gold_label"] = ex["expert_eligibility"]
        pred["gold_evidence"] = parse_expert_sentences(ex["expert_sentences"])
        pred["criterion_id"] = build_criterion_id(
            ex["trial_id"], ex["criterion_type"], ex["annotation_id"]
        )
        pred["annotation_id"] = ex["annotation_id"]
        pred["patient_id"] = ex["patient_id"]

        predictions.append(pred)

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / f"{split}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {pred_path}")

    return predictions


if __name__ == "__main__":
    # Quick test
    adapter_path = "task1/outputs/adapter"
    if Path(adapter_path).exists():
        pipeline = EligibilityPipeline(adapter_path)
        # Test with sample
        test_note = "0. Patient is a 65-year-old male with hypertension.\n1. No history of diabetes."
        result = pipeline.predict(test_note, "inclusion", "Age >= 18")
        print(json.dumps(result, indent=2))
    else:
        print(f"Adapter not found at {adapter_path}")