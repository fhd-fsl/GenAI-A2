"""Tokenization and preprocessing for TrialGPT criterion-level eligibility fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import Dataset, DatasetDict
from datasets import load_from_disk
from transformers import AutoTokenizer

from data.dataset_interface import TrialGPTDataset, SplitName

# Constants
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 4096
SYSTEM_PROMPT = (
    "You are a clinical trial eligibility expert. Given a patient description, "
    "a criterion type (inclusion/exclusion), and the criterion text, determine "
    "the patient's eligibility for that criterion. Return a JSON object with: "
    "prediction (one of: included, not included, not excluded, excluded, "
    "not enough information, not applicable), evidence_sentence_ids (list of "
    "sentence indices from the patient note that support your decision), "
    "evidence_text (the corresponding sentence texts), and explanation (your "
    "reasoning based only on the provided information)."
)

# Valid eligibility labels (preserve all 6 per assignment §47)
ELIGIBILITY_LABELS = [
    "included",
    "not included",
    "not excluded",
    "excluded",
    "not enough information",
    "not applicable",
]


@dataclass(frozen=True)
class TokenizedExample:
    """Single tokenized training example."""
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    # Metadata for logging/debugging
    annotation_id: int
    patient_id: str
    criterion_id: str
    truncated: bool
    tokens_removed: int


def parse_expert_sentences(sentences_str: str) -> list[int]:
    """Parse expert_sentences string like '[0, 1, 2]' into list of ints."""
    stripped = sentences_str.strip()
    if not stripped or stripped == "[]":
        return []
    # Remove brackets and split by comma
    return [int(x.strip()) for x in stripped[1:-1].split(",")]


def split_note_into_sentences(note: str) -> list[str]:
    """Split patient note into sentences by numbered lines (0., 1., 2...)."""
    lines = note.strip().split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and ". " in line[:5]:
            # Format: "0. text" or "12. text"
            _, text = line.split(". ", 1)
            sentences.append(text.strip())
        elif line:
            # Fallback: treat as continuation or unnumbered
            sentences.append(line)
    return sentences


def build_criterion_id(trial_id: str, criterion_type: str, annotation_id: int) -> str:
    """Construct unique criterion identifier."""
    type_prefix = "I" if criterion_type == "inclusion" else "E"
    return f"{trial_id}_{type_prefix}_{annotation_id}"


def build_user_prompt(note: str, criterion_type: str, criterion_text: str) -> str:
    """Build the user prompt with patient note and criterion."""
    return (
        f"Patient Description:\n{note}\n\n"
        f"Criterion Type: {criterion_type}\n"
        f"Criterion Text: {criterion_text}"
    )


def build_target_json(
    criterion_id: str,
    criterion_type: str,
    criterion_text: str,
    expert_eligibility: str,
    expert_sentences: str,
    note: str,
) -> str:
    """Build the target JSON string for training."""
    sentence_ids = parse_expert_sentences(expert_sentences)
    sentences = split_note_into_sentences(note)
    evidence_text = [sentences[i] for i in sentence_ids if i < len(sentences)]

    target = {
        "criterion_id": criterion_id,
        "criterion_type": criterion_type,
        "criterion_text": criterion_text,
        "prediction": expert_eligibility,
        "evidence_sentence_ids": sentence_ids,
        "evidence_text": evidence_text,
        "explanation": "",  # Model generates this; empty for training target
    }
    import json
    return json.dumps(target, ensure_ascii=False)


def truncate_patient_note(
    tokenizer: AutoTokenizer,
    note: str,
    criterion_type: str,
    criterion_text: str,
    max_tokens: int = MAX_SEQ_LEN,
) -> tuple[str, bool, int]:
    """Truncate patient note from the END if total exceeds max_tokens.

    Returns:
        (truncated_note, was_truncated, tokens_removed)
    """
    # Build full conversation to measure tokens
    user_prompt = build_user_prompt(note, criterion_type, criterion_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # Apply chat template without tokenizing to get string
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Check if over limit
    token_count = len(tokenizer(full_prompt).input_ids)
    if token_count <= max_tokens:
        return note, False, 0

    # Truncate note from END (last sentences first)
    sentences = split_note_into_sentences(note)
    original_count = token_count
    removed = 0

    while token_count > max_tokens and len(sentences) > 1:
        sentences.pop()  # Remove last sentence
        truncated_note = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
        user_prompt = build_user_prompt(truncated_note, criterion_type, criterion_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_count = len(tokenizer(full_prompt).input_ids)
        removed += 1

    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences)), True, removed


def tokenize_example(
    tokenizer: AutoTokenizer,
    example: dict,
) -> TokenizedExample:
    """Tokenize a single dataset example for causal LM training."""
    note = example["note"]
    criterion_type = example["criterion_type"]
    criterion_text = example["criterion_text"]
    expert_eligibility = example["expert_eligibility"]
    expert_sentences = example["expert_sentences"]
    annotation_id = example["annotation_id"]
    patient_id = example["patient_id"]
    trial_id = example["trial_id"]

    criterion_id = build_criterion_id(trial_id, criterion_type, annotation_id)

    # Truncate patient note if needed
    truncated_note, was_truncated, tokens_removed = truncate_patient_note(
        tokenizer, note, criterion_type, criterion_text
    )

    # Build messages
    user_prompt = build_user_prompt(truncated_note, criterion_type, criterion_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Target JSON (what model should generate)
    target_json = build_target_json(
        criterion_id, criterion_type, criterion_text,
        expert_eligibility, expert_sentences, truncated_note
    )

    # Apply chat template with target
    full_messages = messages + [{"role": "assistant", "content": target_json}]
    tokenized = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Find where assistant response starts (after the generation prompt)
    # For Qwen3, the assistant header is "<|im_start|>assistant\n"
    assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    header_len = len(assistant_header)

    # Find the start of assistant content in input_ids
    assistant_start = -1
    for i in range(len(input_ids) - header_len):
        if input_ids[i:i+header_len] == assistant_header:
            assistant_start = i + header_len
            break

    if assistant_start == -1:
        # Fallback: assume last 1/3 is assistant (shouldn't happen with proper template)
        assistant_start = len(input_ids) * 2 // 3

    # Labels: -100 for prompt, actual tokens for assistant response
    labels = [-100] * assistant_start + input_ids[assistant_start:]

    # Ensure lengths match
    assert len(input_ids) == len(labels) == len(attention_mask)

    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        annotation_id=annotation_id,
        patient_id=patient_id,
        criterion_id=criterion_id,
        truncated=was_truncated,
        tokens_removed=tokens_removed,
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    split_name: SplitName,
) -> tuple[Dataset, list[dict]]:
    """Tokenize an entire dataset split.

    Returns:
        (tokenized_dataset, truncation_log)
    """
    tokenized_examples = []
    truncation_log = []

    for example in dataset:
        tok_ex = tokenize_example(tokenizer, example)
        tokenized_examples.append({
            "input_ids": tok_ex.input_ids,
            "attention_mask": tok_ex.attention_mask,
            "labels": tok_ex.labels,
        })
        if tok_ex.truncated:
            truncation_log.append({
                "split": split_name,
                "annotation_id": tok_ex.annotation_id,
                "patient_id": tok_ex.patient_id,
                "criterion_id": tok_ex.criterion_id,
                "tokens_removed": tok_ex.tokens_removed,
            })

    # Convert to Dataset
    tokenized_ds = Dataset.from_list(tokenized_examples)

    return tokenized_ds, truncation_log


TOKENIZED_DATA_DIR = Path(__file__).parent.parent / "data" / "tokenized"
TRUNCATION_LOG_PATH = TOKENIZED_DATA_DIR / "truncation_log.json"


def save_tokenized_splits(
    tokenized_splits: DatasetDict,
    truncation_log: list[dict],
    output_dir: Path = TOKENIZED_DATA_DIR,
) -> None:
    """Save tokenized splits and truncation log to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenized_splits.save_to_disk(str(output_dir))
    import json
    with open(TRUNCATION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(truncation_log, f, indent=2)
    print(f"Tokenized splits saved to {output_dir}")
    print(f"Truncation log saved to {TRUNCATION_LOG_PATH}")


def load_tokenized_splits(
    input_dir: Path = TOKENIZED_DATA_DIR,
) -> tuple[DatasetDict, list[dict]]:
    """Load tokenized splits and truncation log from disk."""
    import json
    tokenized_splits = load_from_disk(str(input_dir))
    with open(TRUNCATION_LOG_PATH, "r", encoding="utf-8") as f:
        truncation_log = json.load(f)
    return tokenized_splits, truncation_log


def prepare_tokenized_splits(
    tokenizer: AutoTokenizer | None = None,
    save: bool = True,
) -> tuple[DatasetDict, list[dict]]:
    """Prepare tokenized train/validation/test splits.

    Returns:
        (tokenized DatasetDict, combined truncation_log)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    dataset = TrialGPTDataset()
    all_splits_dict: dict[str, Dataset] = {}
    all_truncation_log: list[dict] = []

    for split_name in ["train", "validation", "test"]:
        print(f"Tokenizing {split_name} split...")
        split_ds = dataset.get_split(split_name)
        tokenized_split, trunc_log = tokenize_dataset(split_ds, tokenizer, split_name)
        all_splits_dict[split_name] = tokenized_split
        all_truncation_log.extend(trunc_log)

    all_splits = DatasetDict(all_splits_dict)
    print(f"Total truncated examples: {len(all_truncation_log)}")

    if save:
        save_tokenized_splits(all_splits, all_truncation_log)

    return all_splits, all_truncation_log


def main() -> None:
    """Test tokenization on a few examples and save tokenized splits."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = TrialGPTDataset()

    # Test on train split
    train_ds = dataset.get_split("train")
    print(f"Train examples: {len(train_ds)}")

    # Tokenize first 3 examples
    for i in range(3):
        ex = train_ds[i]
        tok_ex = tokenize_example(tokenizer, ex)
        print(f"\n--- Example {i} ---")
        print(f"Patient: {tok_ex.patient_id}")
        print(f"Criterion: {tok_ex.criterion_id}")
        print(f"Truncated: {tok_ex.truncated} (removed {tok_ex.tokens_removed} tokens)")
        print(f"Input length: {len(tok_ex.input_ids)}")
        print(f"Label length: {len(tok_ex.labels)}")

        # Decode assistant portion
        assistant_start = tok_ex.labels.index(next(x for x in tok_ex.labels if x != -100))
        decoded = tokenizer.decode(tok_ex.input_ids[assistant_start:], skip_special_tokens=True)
        print(f"Target JSON:\n{decoded[:500]}...")

    # Full tokenization + save
    print("\n--- Full tokenization ---")
    tokenized_splits, trunc_log = prepare_tokenized_splits(tokenizer, save=True)
    for split_name, ds in tokenized_splits.items():
        lengths = [len(ex["input_ids"]) for ex in ds]
        print(f"{split_name}: {len(ds)} examples, "
              f"len: min={min(lengths)}, max={max(lengths)}, "
              f"avg={np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()