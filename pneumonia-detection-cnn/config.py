"""
config.py
=========
Single source of truth for every path, hyperparameter, and constant used
across training, evaluation, and the Streamlit app. Nothing else in the
project should hard-code a magic number — import it from here instead.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# Root of the Kaggle "Chest X-Ray Images (Pneumonia)" dataset.
# Expected structure:
#   DATA_DIR/train/{NORMAL,PNEUMONIA}
#   DATA_DIR/val/{NORMAL,PNEUMONIA}
#   DATA_DIR/test/{NORMAL,PNEUMONIA}
DATA_DIR = Path(os.environ.get("PNEUMONIA_DATA_DIR", ROOT_DIR / "data" / "chest_xray"))
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"          # metrics.json, history.json, plots
SAMPLE_IMAGES_DIR = ROOT_DIR / "sample_images"  # a few demo images shipped with the repo

BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"
HISTORY_PATH = ARTIFACTS_DIR / "history.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = ARTIFACTS_DIR / "roc_curve.png"
PR_CURVE_PATH = ARTIFACTS_DIR / "pr_curve.png"
TRAINING_CURVES_PATH = ARTIFACTS_DIR / "training_curves.png"

for d in (MODELS_DIR, ARTIFACTS_DIR, SAMPLE_IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Class labels
# ----------------------------------------------------------------------
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # index 0 / 1 -> must match generator's class_indices
POSITIVE_CLASS = "PNEUMONIA"

# ----------------------------------------------------------------------
# Image / data hyperparameters
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)          # works for both the custom CNN and transfer-learning backbones
IMG_CHANNELS = 3               # chest X-rays are grayscale but we replicate to 3 ch for transfer learning
INPUT_SHAPE = (*IMG_SIZE, IMG_CHANNELS)
BATCH_SIZE = 32
SEED = 42

# ----------------------------------------------------------------------
# Model / architecture selection
# ----------------------------------------------------------------------
# "custom_cnn"   -> hand-built residual + squeeze-excite CNN (see model_builder.py)
# "efficientnet" -> EfficientNetB0 transfer learning
# "densenet"     -> DenseNet121 transfer learning
# "resnet"       -> ResNet50V2 transfer learning
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "efficientnet")

DROPOUT_RATE = 0.35
L2_REG = 1e-4
LABEL_SMOOTHING = 0.05

# ----------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------
EPOCHS_HEAD = 10          # warm-up epochs training only the classification head (transfer learning)
EPOCHS_FINE_TUNE = 25     # additional epochs with the backbone partially unfrozen
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
FINE_TUNE_AT_LAYER = 100  # unfreeze backbone layers from this index onward
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

USE_MIXED_PRECISION = True
USE_CLASS_WEIGHTS = True
DECISION_THRESHOLD = 0.5   # can be re-tuned in evaluate.py via the ROC curve (Youden's J)

# ----------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------
GRADCAM_ALPHA = 0.45  # heatmap overlay opacity

# ----------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------
APP_TITLE = "PulmoVision AI"
APP_TAGLINE = "Deep-learning-assisted pneumonia screening from chest radiographs"
