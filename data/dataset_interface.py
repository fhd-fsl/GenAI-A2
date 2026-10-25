"""Dataset interface for loading processed TrialGPT splits."""

from pathlib import Path
from typing import Literal

from datasets import Dataset, DatasetDict, load_from_disk

PROCESSED_DATA_DIR = Path(__file__).parent / "processed"
SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "split_manifest.json"

SplitName = Literal["train", "validation", "test"]


class TrialGPTDataset:
    """Interface for accessing the TrialGPT Criterion Annotations dataset splits."""

    def __init__(self, data_dir: Path = PROCESSED_DATA_DIR) -> None:
        """Initialize and load the processed dataset.

        Args:
            data_dir: Directory containing the processed dataset.
        """
        self._data_dir = data_dir
        self._dataset: DatasetDict | None = None
        self._manifest: dict | None = None
        self._load()

    def _load(self) -> None:
        """Load dataset and manifest from disk."""
        self._dataset = load_from_disk(str(self._data_dir))
        import json
        with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            self._manifest = json.load(f)

    def get_split(self, split: SplitName) -> Dataset:
        """Get a dataset split.

        Args:
            split: One of "train", "validation", "test".

        Returns:
            The requested Dataset split.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded")
        return self._dataset[split]

    def get_manifest(self) -> dict:
        """Get the split manifest with patient IDs and counts."""
        if self._manifest is None:
            raise RuntimeError("Manifest not loaded")
        return self._manifest

    def get_patient_ids(self, split: SplitName) -> list[str]:
        """Get patient IDs for a specific split."""
        return self._manifest[split]["patient_ids"]

    def get_stats(self) -> dict:
        """Get summary statistics for all splits."""
        return {
            split: {
                "n_patients": self._manifest[split]["n_patients"],
                "n_annotations": self._manifest[split]["n_annotations"],
            }
            for split in ["train", "validation", "test"]
        }

    def print_summary(self) -> None:
        """Print a summary of the dataset splits."""
        stats = self.get_stats()
        print("TrialGPT Criterion Annotations - Processed Splits")
        print("=" * 50)
        print(f"Split seed: {self._manifest['split_seed']}")
        print()
        for split in ["train", "validation", "test"]:
            s = stats[split]
            print(f"  {split.capitalize():12s}: {s['n_patients']:2d} patients, {s['n_annotations']:4d} annotations")
        print()
        total_patients = sum(s["n_patients"] for s in stats.values())
        total_annotations = sum(s["n_annotations"] for s in stats.values())
        print(f"  {'Total':12s}: {total_patients:2d} patients, {total_annotations:4d} annotations")


def load_dataset_split(split: SplitName, data_dir: Path = PROCESSED_DATA_DIR) -> Dataset:
    """Convenience function to load a single split.

    Args:
        split: One of "train", "validation", "test".
        data_dir: Directory containing the processed dataset.

    Returns:
        The requested Dataset split.
    """
    return TrialGPTDataset(data_dir).get_split(split)


def load_manifest(data_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """Load the split manifest."""
    return TrialGPTDataset(data_dir).get_manifest()


def main() -> None:
    """Print dataset summary when run as script."""
    TrialGPTDataset().print_summary()


if __name__ == "__main__":
    main()