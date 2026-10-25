"""Download and cache the TrialGPT Criterion Annotations dataset locally."""

from pathlib import Path
from datasets import load_dataset, DatasetDict


DATASET_NAME = "ncbi/TrialGPT-Criterion-Annotations"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "raw"


def download_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> DatasetDict:
    """Download the dataset and save to disk in Hugging Face format.

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        The loaded DatasetDict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)

    print(f"Saving to {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    print("Done.")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Features: {list(dataset['train'].features.keys())}")

    return dataset


def main() -> None:
    """Entry point for script execution."""
    download_dataset()


if __name__ == "__main__":
    main()