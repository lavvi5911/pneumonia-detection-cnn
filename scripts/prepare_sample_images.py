"""
scripts/prepare_sample_images.py
==================================
Copies a small handful of test-set images (a mix of NORMAL and PNEUMONIA)
into sample_images/, so the Streamlit app's "try a sample image" dropdown
works out of the box for reviewers who don't want to source their own
chest X-ray to upload.

Usage:
    python scripts/prepare_sample_images.py --n_per_class 3
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def main(n_per_class: int):
    config.SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in config.CLASS_NAMES:
        class_dir = config.TEST_DIR / class_name
        if not class_dir.exists():
            print(f"[WARN] {class_dir} not found — did you run scripts/download_dataset.py?")
            continue
        files = sorted(class_dir.glob("*"))[:n_per_class]
        for f in files:
            dest = config.SAMPLE_IMAGES_DIR / f"{class_name.lower()}_{f.name}"
            shutil.copy2(f, dest)
            print(f"  copied {f} -> {dest}")
    print(f"[INFO] Sample images ready in {config.SAMPLE_IMAGES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=3)
    args = parser.parse_args()
    main(args.n_per_class)
