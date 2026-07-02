"""
pages/3_How_It_Works.py
=========================
Educational deep-dive into the architecture, training pipeline, and
explainability method — the "show your work" page.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src import app_helpers as ui

ui.page_setup("How It Works", page_icon="🧠")
ui.render_hero(
    "How It Works",
    "A guided tour of the data pipeline, model architecture, training strategy, and the "
    "Grad-CAM explainability method behind every prediction.",
    badge_text="Under the hood",
)

tab_data, tab_model, tab_train, tab_xai = st.tabs(
    ["📁 Data Pipeline", "🏗️ Architecture", "🎯 Training Strategy", "🔥 Explainability"]
)

with tab_data:
    st.markdown("#### From raw JPEGs to model-ready tensors")
    st.markdown(
        f"""
- **Source:** the Kaggle *Chest X-Ray Images (Pneumonia)* dataset, pre-split into
  `train/`, `val/`, and `test/` folders, each containing `NORMAL/` and `PNEUMONIA/` sub-folders.
- **Loading:** a `tf.data.Dataset` pipeline reads file paths, decodes JPEG/PNG bytes, resizes to
  **{config.IMG_SIZE[0]}×{config.IMG_SIZE[1]}**, and normalizes pixel values to **[0, 1]**.
- **Augmentation** (train split only): small random rotations, translations, zoom, brightness,
  and contrast jitter — deliberately *no vertical flips*, since chest X-rays have a fixed
  anatomical orientation that a flipped image would violate.
- **Class imbalance:** the raw dataset has roughly 3× more PNEUMONIA than NORMAL images.
  We compute **inverse-frequency class weights** so the loss function penalizes mistakes on the
  minority (NORMAL) class more heavily, rather than letting the model default to "always predict
  pneumonia".
- **Performance:** `.cache()` + `.prefetch(AUTOTUNE)` keep the GPU fed without I/O stalls.
        """
    )

with tab_model:
    st.markdown("#### Two architecture families, one interface")
    st.markdown(
        """
**1. PulmoNet (custom residual CNN)** — built from scratch to showcase CNN internals end-to-end:
        """
    )
    ui.render_step("Stem", "7×7 conv, stride 2 → BatchNorm → ReLU → 3×3 max-pool",
                    "Aggressively downsamples the input before the expensive residual stages.")
    ui.render_step("Residual stages", "4 stages, 64→128→256→512 filters, pre-activation blocks",
                    "Each block is BN→ReLU→Conv ×2 with a projection shortcut, easing gradient flow in a deep network.")
    ui.render_step("Squeeze-and-Excitation", "Channel attention after every residual block",
                    "Global-average-pools each block's feature map, learns per-channel importance weights via a "
                    "tiny 2-layer MLP, and rescales the channels — letting the network emphasize informative "
                    "feature maps (e.g. texture patterns associated with lung opacities).")
    ui.render_step("Head", "Global average pool → Dense(256) → Dropout → Dense(1, sigmoid)",
                    "GAP instead of Flatten drastically cuts parameters and reduces overfitting risk.")

    st.markdown("**2. Transfer learning backbones** (EfficientNetB0 / DenseNet121 / ResNet50V2)")
    st.markdown(
        """
ImageNet-pretrained convolutional backbones already encode powerful general-purpose visual
features (edges, textures, shapes). We attach the same classification head described above and
fine-tune in two phases (see the Training Strategy tab) — this typically reaches higher accuracy
with less training data and time than training a CNN from scratch.
        """
    )

with tab_train:
    st.markdown("#### Two-phase transfer learning schedule")
    ui.render_step("Phase 1 — Head warm-up", f"{config.EPOCHS_HEAD} epochs, backbone frozen, lr={config.LEARNING_RATE_HEAD}",
                    "Only the newly-initialized dense head trains, so its random weights don't send large, "
                    "destructive gradients back through the pretrained backbone.")
    ui.render_step("Phase 2 — Fine-tuning", f"up to {config.EPOCHS_FINE_TUNE} epochs, "
                    f"top layers unfrozen (from layer {config.FINE_TUNE_AT_LAYER}), lr={config.LEARNING_RATE_FINE_TUNE}",
                    "Backbone layers from the given index onward become trainable at a much lower learning "
                    "rate, adapting high-level features to chest-X-ray-specific patterns without catastrophic forgetting.")

    st.markdown("#### Regularization & optimization tricks")
    st.markdown(
        f"""
- **Class-weighted binary cross-entropy** with **label smoothing** ({config.LABEL_SMOOTHING}) to reduce overconfidence
- **L2 weight decay** ({config.L2_REG}) on convolutional/dense kernels
- **Dropout** ({config.DROPOUT_RATE}) before the final classification layers
- **Mixed precision (float16)** training for ~2× speedup on modern GPUs, with a float32 output head for numerical stability
- **EarlyStopping** on validation AUC (patience {config.EARLY_STOPPING_PATIENCE}) with best-weight restoration
- **ReduceLROnPlateau** on validation loss (factor {config.REDUCE_LR_FACTOR}, patience {config.REDUCE_LR_PATIENCE})
- **ModelCheckpoint** saving only the best validation-AUC epoch, so a late overfitting epoch never overwrites the best model
        """
    )

with tab_xai:
    st.markdown("#### Grad-CAM: seeing what the model sees")
    st.markdown(
        """
Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question: *which pixels
mattered most for this prediction?*

1. Run a forward pass, capturing the activations of the **last convolutional layer** — the
   final spatial feature map before global pooling.
2. Compute the gradient of the predicted class score with respect to those activations.
3. Global-average-pool the gradients across spatial dimensions to get one **importance weight
   per channel** — how much each feature-map channel influenced the prediction.
4. Compute a weighted sum of the activation channels, apply ReLU (we only care about features
   that *positively* influenced the pneumonia score), and normalize to [0, 1].
5. Resize the resulting low-resolution heatmap back up to the original image size and overlay
   it with a "turbo" colormap — warm colors (red/orange) mark high-influence regions.

This is what powers the heatmap overlay on the **Diagnose** page: a sanity check that the model
is actually attending to lung fields and opacities, not to spurious artifacts like text markers
or image borders.
        """
    )

ui.render_footer()
