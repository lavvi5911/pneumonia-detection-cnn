"""
evaluate.py
===========
Runs the trained model against the held-out test split and produces every
artifact the Streamlit "Model Insights" page consumes:

  artifacts/metrics.json          - scalar metrics + classification report
  artifacts/confusion_matrix.png
  artifacts/roc_curve.png
  artifacts/pr_curve.png
  artifacts/training_curves.png   - loss/accuracy/auc curves from history.json

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import config
from src import data_loader, utils

PALETTE = {"bg": "#0B1F3A", "accent": "#14B8A6", "warn": "#F97316", "grid": "#1E3A5F"}


def _style_axes(ax):
    ax.set_facecolor("#0F2947")
    ax.tick_params(colors="#CBD5E1")
    for spine in ax.spines.values():
        spine.set_color("#1E3A5F")
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")


def plot_confusion_matrix(cm: np.ndarray, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.6), facecolor="#0B1F3A")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="mako", cbar=False,
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
        annot_kws={"size": 14, "color": "white"}, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Test Set")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    best_threshold = utils.youden_j_threshold(fpr, tpr, thresholds)

    fig, ax = plt.subplots(figsize=(5.5, 4.6), facecolor="#0B1F3A")
    ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#64748B", lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", facecolor="#0F2947", edgecolor="none", labelcolor="#E2E8F0")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return roc_auc, best_threshold


def plot_pr_curve(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(5.5, 4.6), facecolor="#0B1F3A")
    ax.plot(recall, precision, color=PALETTE["warn"], lw=2.5, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", facecolor="#0F2947", edgecolor="none", labelcolor="#E2E8F0")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return pr_auc


def plot_training_curves(history: dict, out_path):
    metrics_to_plot = [m for m in ("loss", "accuracy", "auc") if m in history]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5.5 * len(metrics_to_plot), 4.2),
                              facecolor="#0B1F3A")
    if len(metrics_to_plot) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics_to_plot):
        ax.plot(history[metric], color=PALETTE["accent"], lw=2, label=f"train_{metric}")
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(history[val_key], color=PALETTE["warn"], lw=2, label=f"val_{metric}")
        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.legend(facecolor="#0F2947", edgecolor="none", labelcolor="#E2E8F0")
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def evaluate(model_path=config.BEST_MODEL_PATH):
    utils.set_global_seed(config.SEED)
    model = tf.keras.models.load_model(model_path)

    test_ds, n_test = data_loader.make_dataset("test", batch_size=config.BATCH_SIZE, shuffle=False, augment=False)

    y_true, y_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).ravel()
        y_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    roc_auc, best_threshold = plot_roc_curve(y_true, y_prob, config.ROC_CURVE_PATH)
    pr_auc = plot_pr_curve(y_true, y_prob, config.PR_CURVE_PATH)

    y_pred_default = (y_prob >= config.DECISION_THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred_default)
    plot_confusion_matrix(cm, config.CONFUSION_MATRIX_PATH)

    report = classification_report(
        y_true, y_pred_default, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics_payload = {
        "n_test": int(n_test),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(report["accuracy"]),
        "sensitivity_recall_positive": float(sensitivity),
        "specificity": float(specificity),
        "precision_positive": float(report[config.POSITIVE_CLASS]["precision"]),
        "recall_positive": float(report[config.POSITIVE_CLASS]["recall"]),
        "f1_positive": float(report[config.POSITIVE_CLASS]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "default_threshold": config.DECISION_THRESHOLD,
        "youden_optimal_threshold": float(best_threshold),
        "architecture": getattr(model, "name", "unknown"),
        "total_params": int(model.count_params()),
    }
    utils.save_json(metrics_payload, config.METRICS_PATH)

    if config.HISTORY_PATH.exists():
        history = utils.load_json(config.HISTORY_PATH)
        plot_training_curves(history, config.TRAINING_CURVES_PATH)

    print("=" * 60)
    print(f"Test accuracy      : {metrics_payload['accuracy']:.4f}")
    print(f"ROC AUC            : {roc_auc:.4f}")
    print(f"PR AUC             : {pr_auc:.4f}")
    print(f"Sensitivity (recall of PNEUMONIA): {sensitivity:.4f}")
    print(f"Specificity        : {specificity:.4f}")
    print(f"Youden-optimal thr : {best_threshold:.4f}")
    print("=" * 60)
    return metrics_payload


if __name__ == "__main__":
    evaluate()
