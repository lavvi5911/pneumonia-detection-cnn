"""
gradcam.py
==========
Grad-CAM (Gradient-weighted Class Activation Mapping) for visual
explainability: highlights which regions of the chest X-ray most
influenced the model's prediction. Critical for a medical-imaging demo —
a black-box "PNEUMONIA: 92%" is far less trustworthy than one with a
heatmap over the actual opacity.
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

import config


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Auto-detects the last 4D-output (spatial) layer to hook Grad-CAM onto.
    Works for both the custom CNN and nested transfer-learning backbones.
    """
    # Search top-level model layers in reverse first.
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except AttributeError:
            continue

    # If the last conv layer lives inside a nested backbone (transfer learning
    # models wrap e.g. EfficientNetB0 as a single sub-model), search inside it.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                try:
                    if len(sub_layer.output_shape) == 4:
                        return f"{layer.name}/{sub_layer.name}"
                except AttributeError:
                    continue

    raise ValueError("Could not find a 4D convolutional layer for Grad-CAM.")


def _resolve_conv_layer(model: tf.keras.Model, layer_name: str):
    """Supports 'outer_layer/inner_layer' addressing for nested backbones."""
    if "/" in layer_name:
        outer_name, inner_name = layer_name.split("/", 1)
        outer = model.get_layer(outer_name)
        return outer.get_layer(inner_name), outer
    return model.get_layer(layer_name), None


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    layer_name: str | None = None,
) -> np.ndarray:
    """Computes a normalized [0, 1] Grad-CAM heatmap for a single preprocessed
    image batch of shape (1, H, W, 3).
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    conv_layer, outer_layer = _resolve_conv_layer(model, layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        # Binary sigmoid output: the "class score" IS the predicted probability.
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # channel importance weights

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = config.GRADCAM_ALPHA,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Overlays a Grad-CAM heatmap onto the original RGB image (both as
    numpy arrays, `original_img` in [0, 255] uint8).
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = heatmap_color.astype(np.float32) * alpha + original_img.astype(np.float32) * (1 - alpha)
    return np.clip(overlaid, 0, 255).astype(np.uint8)


def generate_gradcam_overlay(
    model: tf.keras.Model,
    preprocessed_batch: np.ndarray,
    original_rgb_uint8: np.ndarray,
    layer_name: str | None = None,
) -> np.ndarray:
    """Convenience wrapper: preprocessed_batch -> heatmap -> overlaid image."""
    heatmap = make_gradcam_heatmap(preprocessed_batch, model, layer_name)
    return overlay_heatmap(original_rgb_uint8, heatmap)
