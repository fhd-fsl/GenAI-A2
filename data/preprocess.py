"""Preprocess the TrialGPT dataset: remove unused columns, create patient-grouped split."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

# Constants
RAW_DATA_DIR = Path(__file__).parent / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent / "processed"
SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "split_manifest.json"

# Columns to KEEP for the task
KEEP_COLUMNS = [
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
]

# Columns to REMOVE (GPT-4 predictions, metadata not needed for core task)
REMOVE_COLUMNS = [
    "trial_title",
    "gpt4_explanation",
    "explanation_correctness",
    "gpt4_sentences",
    "gpt4_eligibility",
    "training",
]

# Split configuration
SPLIT_SEED = 42
N_TRAIN_PATIENTS = 37
N_VAL_PATIENTS = 8
N_TEST_PATIENTS = 8

SplitName = Literal["train", "validation", "test"]


def load_raw_dataset(data_dir: Path = RAW_DATA_DIR) -> Dataset:
    """Load the raw dataset from disk."""
    return load_from_disk(str(data_dir))["train"]


def remove_unused_columns(dataset: Dataset) -> Dataset:
    """Remove columns not needed for fine-tuning or evaluation."""
    existing_remove = [c for c in REMOVE_COLUMNS if c in dataset.column_names]
    return dataset.remove_columns(existing_remove)


def create_patient_split(
    dataset: Dataset,
    seed: int = SPLIT_SEED,
    n_train: int = N_TRAIN_PATIENTS,
    n_val: int = N_VAL_PATIENTS,
    n_test: int = N_TEST_PATIENTS,
) -> tuple[list[str], list[str], list[str]]:
    """Create patient-grouped split using numpy default_rng(seed).

    Args:
        dataset: The dataset containing patient_id column.
        seed: Random seed for reproducibility.
        n_train: Number of patients in training set.
        n_val: Number of patients in validation set.
        n_test: Number of patients in test set.

    Returns:
        Tuple of (train_patient_ids, val_patient_ids, test_patient_ids).
    """
    unique_patients = sorted(dataset.unique("patient_id"))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_patients).tolist()

    train_patients = shuffled[:n_train]
    val_patients = shuffled[n_train:n_train + n_val]
    test_patients = shuffled[n_train + n_val:n_train + n_val + n_test]

    return train_patients, val_patients, test_patients


def apply_split(
    dataset: Dataset,
    train_patients: list[str],
    val_patients: list[str],
    test_patients: list[str],
) -> DatasetDict:
    """Split dataset by patient ID into train/validation/test."""
    train_ds = dataset.filter(lambda x: x["patient_id"] in train_patients)
    val_ds = dataset.filter(lambda x: x["patient_id"] in val_patients)
    test_ds = dataset.filter(lambda x: x["patient_id"] in test_patients)

    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })


def build_split_manifest(
    dataset: Dataset,
    train_patients: list[str],
    val_patients: list[str],
    test_patients: list[str],
) -> dict:
    """Build the split manifest with patient IDs and annotation counts."""
    df = dataset.to_pandas()

    def count_annotations(patient_ids: list[str]) -> int:
        return int(df[df["patient_id"].isin(patient_ids)].shape[0])

    manifest = {
        "split_seed": SPLIT_SEED,
        "n_train_patients": N_TRAIN_PATIENTS,
        "n_val_patients": N_VAL_PATIENTS,
        "n_test_patients": N_TEST_PATIENTS,
        "train": {
            "patient_ids": train_patients,
            "n_patients": len(train_patients),
            "n_annotations": count_annotations(train_patients),
        },
        "validation": {
            "patient_ids": val_patients,
            "n_patients": len(val_patients),
            "n_annotations": count_annotations(val_patients),
        },
        "test": {
            "patient_ids": test_patients,
            "n_patients": len(test_patients),
            "n_annotations": count_annotations(test_patients),
        },
    }
    return manifest


def save_split_manifest(manifest: dict, path: Path = SPLIT_MANIFEST_PATH) -> None:
    """Save split manifest as JSON."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(manifest, f, indent=2)
    print(f"Split manifest saved to {path}")


def save_processed_dataset(dataset_dict: DatasetDict, output_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Save processed dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"Processed dataset saved to {output_dir}")
    for split_name, split_ds in dataset_dict.items():
        print(f"  {split_name}: {len(split_ds)} annotations")


def preprocess(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    manifest_path: Path = SPLIT_MANIFEST_PATH,
) -> tuple[DatasetDict, dict]:
    """Run full preprocessing pipeline.

    Returns:
        Tuple of (processed DatasetDict, split manifest dict).
    """
    print("Loading raw dataset...")
    dataset = load_raw_dataset(raw_data_dir)

    print("Removing unused columns...")
    dataset = remove_unused_columns(dataset)

    print("Creating patient-grouped split...")
    train_patients, val_patients, test_patients = create_patient_split(dataset)

    print("Applying split...")
    dataset_dict = apply_split(dataset, train_patients, val_patients, test_patients)

    print("Building split manifest...")
    manifest = build_split_manifest(dataset, train_patients, val_patients, test_patients)

    print("Saving...")
    save_split_manifest(manifest, manifest_path)
    save_processed_dataset(dataset_dict, processed_data_dir)

    return dataset_dict, manifest


def main() -> None:
    """Entry point for script execution."""
    preprocess()


if __name__ == "__main__":
    main()