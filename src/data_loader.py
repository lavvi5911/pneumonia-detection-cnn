"""
data_loader.py
===============
Builds high-performance tf.data pipelines for the train/val/test splits,
with on-the-fly augmentation, class-imbalance-aware sample weighting, and
CLAHE-style contrast enhancement tuned for radiographs.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config


AUTOTUNE = tf.data.AUTOTUNE


# ------------------------------------------------------------------
# Augmentation layers (only active during training, via `training=True`)
# ------------------------------------------------------------------
def build_augmentation_pipeline() -> tf.keras.Sequential:
    """Domain-appropriate augmentations for chest X-rays.

    We deliberately avoid vertical flips and heavy color jitter — X-rays
    are grayscale and anatomically oriented, so those transforms would
    generate physiologically implausible images.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomZoom(0.12, fill_mode="constant", fill_value=0.0),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.12, value_range=(0.0, 1.0)),
        ],
        name="xray_augmentation",
    )


def _list_files_and_labels(split_dir: Path) -> tuple[list[str], list[int]]:
    filepaths, labels = [], []
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = split_dir / class_name
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for f in glob.glob(str(class_dir / ext)):
                filepaths.append(f)
                labels.append(label_idx)
    return filepaths, labels


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, config.IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_dataset(
    split: str,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = False,
    augment: bool = False,
    cache: bool = True,
) -> tuple[tf.data.Dataset, int]:
    """Build a tf.data.Dataset for a given split ('train' | 'val' | 'test').

    Returns (dataset, num_samples).
    """
    split_dir = {"train": config.TRAIN_DIR, "val": config.VAL_DIR, "test": config.TEST_DIR}[split]
    filepaths, labels = _list_files_and_labels(split_dir)
    if len(filepaths) == 0:
        raise FileNotFoundError(
            f"No images found under {split_dir}. Expected sub-folders "
            f"{config.CLASS_NAMES} each containing .jpeg/.jpg/.png files."
        )

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=config.SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    if augment:
        aug = build_augmentation_pipeline()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, len(filepaths)


def compute_class_weights() -> dict:
    """Inverse-frequency class weights to counter the NORMAL/PNEUMONIA imbalance
    (the Kaggle dataset has roughly 3x more PNEUMONIA than NORMAL images).
    """
    _, labels = _list_files_and_labels(config.TRAIN_DIR)
    labels = np.array(labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return {int(c): float(w) for c, w in zip(np.unique(labels), weights)}


def dataset_class_counts(split: str) -> dict:
    split_dir = {"train": config.TRAIN_DIR, "val": config.VAL_DIR, "test": config.TEST_DIR}[split]
    counts = {}
    for class_name in config.CLASS_NAMES:
        class_dir = split_dir / class_name
        n = sum(len(glob.glob(str(class_dir / ext))) for ext in ("*.jpeg", "*.jpg", "*.png"))
        counts[class_name] = n
    return counts


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        try:
            print(split, dataset_class_counts(split))
        except Exception as e:  # pragma: no cover
            print(split, "error:", e)
