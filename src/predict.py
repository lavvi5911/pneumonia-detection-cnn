"""
predict.py
==========
Single-image inference utilities shared by the Streamlit app and any
batch-scoring scripts. Keeps the exact same resize/normalize logic used
during training so there's no train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image

import config


@dataclass
class PredictionResult:
    label: str
    probability_pneumonia: float
    probability_normal: float
    confidence: float
    risk_level: str


def load_and_preprocess_image(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Returns (model_input_batch[1,H,W,3] float32 in [0,1], original_rgb_uint8[H,W,3])
    resized to the model's expected input size (needed so Grad-CAM overlays align).
    """
    rgb = pil_image.convert("RGB").resize(config.IMG_SIZE, Image.BILINEAR)
    arr = np.array(rgb).astype(np.uint8)
    model_input = arr.astype(np.float32) / 255.0
    return np.expand_dims(model_input, axis=0), arr


def classify_risk(probability_pneumonia: float) -> str:
    if probability_pneumonia < 0.3:
        return "Low"
    if probability_pneumonia < 0.6:
        return "Moderate"
    if probability_pneumonia < 0.85:
        return "High"
    return "Very High"


def predict_single(model: tf.keras.Model, pil_image: Image.Image, threshold: float = config.DECISION_THRESHOLD):
    batch, original_rgb = load_and_preprocess_image(pil_image)
    prob_pneumonia = float(model.predict(batch, verbose=0).ravel()[0])
    prob_normal = 1.0 - prob_pneumonia

    label = config.POSITIVE_CLASS if prob_pneumonia >= threshold else "NORMAL"
    confidence = prob_pneumonia if label == config.POSITIVE_CLASS else prob_normal

    result = PredictionResult(
        label=label,
        probability_pneumonia=prob_pneumonia,
        probability_normal=prob_normal,
        confidence=confidence,
        risk_level=classify_risk(prob_pneumonia),
    )
    return result, batch, original_rgb
