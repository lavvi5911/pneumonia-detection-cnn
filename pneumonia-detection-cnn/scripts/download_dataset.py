"""
scripts/download_dataset.py
============================
Downloads the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle and
places it at the path config.DATA_DIR expects (data/chest_xray/{train,val,test}).

Requires a Kaggle account + API token (~/.kaggle/kaggle.json), or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables.

Usage:
    python scripts/download_dataset.py
"""

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def main():
    try:
        import kagglehub
    except ImportError:
        print("[ERROR] kagglehub is not installed. Run: pip install kagglehub")
        sys.exit(1)

    print("[INFO] Downloading paultimothymooney/chest-xray-pneumonia from Kaggle...")
    download_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"[INFO] Downloaded to cache: {download_path}")

    # The Kaggle archive nests the real data one level deeper (chest_xray/chest_xray/...).
    src_candidates = list(Path(download_path).rglob("chest_xray"))
    if not src_candidates:
        print(f"[ERROR] Could not locate a 'chest_xray' folder inside {download_path}. "
              "Inspect the download manually and copy it to data/chest_xray.")
        sys.exit(1)

    # Prefer the deepest match that directly contains train/val/test.
    src = None
    for candidate in src_candidates:
        if (candidate / "train").exists() and (candidate / "test").exists():
            src = candidate
            break
    src = src or src_candidates[0]

    dest = config.DATA_DIR
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[INFO] {dest} already exists — skipping copy. Delete it first to re-copy.")
    else:
        print(f"[INFO] Copying {src} -> {dest} ...")
        shutil.copytree(src, dest)
        print("[INFO] Done.")

    for split in ("train", "val", "test"):
        split_dir = dest / split
        if split_dir.exists():
            n_normal = len(list((split_dir / "NORMAL").glob("*")))
            n_pneumonia = len(list((split_dir / "PNEUMONIA").glob("*")))
            print(f"  {split:5s}: NORMAL={n_normal:5d}  PNEUMONIA={n_pneumonia:5d}")
        else:
            print(f"  [WARN] {split_dir} missing.")


if __name__ == "__main__":
    main()
