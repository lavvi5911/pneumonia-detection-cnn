"""
utils.py
========
Small, dependency-light helpers shared across train/evaluate/app so none
of those scripts have to duplicate boilerplate.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Best-effort determinism across numpy / tensorflow / python's random."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def enable_mixed_precision() -> None:
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (mixed_float16).")
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Could not enable mixed precision: {e}")


def configure_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"[INFO] {len(gpus)} GPU(s) detected." if gpus else "[INFO] No GPU detected — running on CPU.")


def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def youden_j_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    """Finds the classification threshold that maximizes sensitivity + specificity - 1
    (Youden's J statistic), often a better operating point than the default 0.5
    for imbalanced medical classification tasks.
    """
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def count_params(model: tf.keras.Model) -> dict:
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_trainable = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}
