"""Helper script to download fine-tuned LoRA adapter weights from Google Drive.

Usage:
    python download_adapter_weights.py
"""

import sys
from pathlib import Path
import urllib.request

ADAPTER_DIR = Path(__file__).parent / "task1" / "outputs" / "adapter"
WEIGHTS_FILE = ADAPTER_DIR / "adapter_model.safetensors"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1YEafpplZJwsxZx9rfEQxkuEPuCg8xUrp?usp=sharing"

# Direct file ID for adapter_model.safetensors if shared
# Or direct gdown instruction
def main():
    print("=" * 70)
    print("  TrialMatch: LoRA Adapter Weights Downloader")
    print("=" * 70)

    if WEIGHTS_FILE.exists():
        size_mb = round(WEIGHTS_FILE.stat().st_size / (1024 * 1024), 1)
        print(f"\n[OK] Adapter weights already exist at:")
        print(f"     {WEIGHTS_FILE} ({size_mb} MB)")
        print("     No download needed!\n")
        return

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nAdapter weights must be placed at:")
    print(f"  {WEIGHTS_FILE}")
    print(f"\nGoogle Drive Folder:")
    print(f"  {DRIVE_FOLDER_URL}")
    print("\nAttempting automatic download via gdown (if installed)...")

    try:
        import gdown
        # Download from folder
        url = DRIVE_FOLDER_URL
        gdown.download_folder(url, output=str(ADAPTER_DIR), quiet=False, use_cookies=False)
        if WEIGHTS_FILE.exists():
            print("\n[SUCCESS] Successfully downloaded adapter weights!")
            return
    except ImportError:
        print("[INFO] 'gdown' is not installed.")
    except Exception as e:
        print(f"[!] Automated download encountered: {e}")

    print("\n" + "=" * 70)
    print("  MANUAL DOWNLOAD INSTRUCTIONS:")
    print("  1. Open the Google Drive folder in your browser:")
    print(f"     {DRIVE_FOLDER_URL}")
    print("  2. Download 'adapter_model.safetensors' (479 MB)")
    print(f"  3. Save it into:")
    print(f"     {ADAPTER_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

