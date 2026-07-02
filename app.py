"""
app.py
======
Home / landing page for PulmoVision AI. Run with:

    streamlit run app.py

Other pages (Diagnose, Model Insights, How It Works, About) live in pages/
and are picked up automatically by Streamlit's multipage navigation.
"""

import streamlit as st

import config
from src import app_helpers as ui

ui.page_setup("Home", page_icon="🫁")

model_ready = ui.model_available()
badge = "Model online · ready to diagnose" if model_ready else "Demo mode · no trained model found yet"
ui.render_hero(config.APP_TITLE, config.APP_TAGLINE, badge_text=badge)

if not model_ready:
    st.warning(
        "No trained model was found in `models/`. The **Diagnose** page will not be able to run "
        "predictions until you train a model (`python -m src.train`) or place a `best_model.keras` "
        "file in the `models/` folder. Everything else in the app still works.",
        icon="⚠️",
    )

# ------------------------------------------------------------------
# Key stats
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Architecture", config.MODEL_ARCHITECTURE.replace("_", " ").title())
with col2:
    st.metric("Input resolution", f"{config.IMG_SIZE[0]}×{config.IMG_SIZE[1]}")
with col3:
    st.metric("Classes", "2 (Normal / Pneumonia)")
with col4:
    metrics_path = config.METRICS_PATH
    if metrics_path.exists():
        import json
        m = json.loads(metrics_path.read_text())
        st.metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
    else:
        st.metric("Test ROC-AUC", "—")

ui.render_divider()

# ------------------------------------------------------------------
# What this app does
# ------------------------------------------------------------------
st.markdown("### What PulmoVision AI does")
c1, c2, c3 = st.columns(3)
with c1:
    ui.render_card(
        "Upload & Diagnose",
        "Upload a chest X-ray (JPEG/PNG) and get an instant NORMAL / PNEUMONIA "
        "prediction with a calibrated confidence score.",
        icon="🩻",
    )
with c2:
    ui.render_card(
        "Grad-CAM Explainability",
        "See exactly which lung regions drove the model's decision via a "
        "gradient-based class activation heatmap overlaid on the scan.",
        icon="🔥",
    )
with c3:
    ui.render_card(
        "Transparent Model Insights",
        "Inspect ROC/PR curves, the confusion matrix, and full training "
        "history for the exact model powering this app.",
        icon="📊",
    )

ui.render_divider()

# ------------------------------------------------------------------
# How it works
# ------------------------------------------------------------------
left, right = st.columns([1.1, 1])
with left:
    st.markdown("### How it works")
    ui.render_step("01", "Upload", "Drop in a chest X-ray image on the Diagnose page — JPEG or PNG.")
    ui.render_step("02", "Preprocess", "The image is resized to "
                   f"{config.IMG_SIZE[0]}×{config.IMG_SIZE[1]} and normalized identically to training.")
    ui.render_step("03", "Inference", "A CNN "
                   f"({config.MODEL_ARCHITECTURE.replace('_', ' ')}) outputs a pneumonia probability.")
    ui.render_step("04", "Explain", "Grad-CAM highlights the pixels that most influenced the prediction.")
    ui.render_step("05", "Review", "You get a verdict, confidence score, risk tier, and heatmap — together.")

with right:
    st.markdown("### Model architecture")
    if config.MODEL_ARCHITECTURE == "custom_cnn":
        st.markdown(
            "**PulmoNet** — a from-scratch residual CNN built for this project:\n\n"
            "- 7×7 stem convolution + max-pool\n"
            "- 4 residual stages (64 → 128 → 256 → 512 filters)\n"
            "- Squeeze-and-Excitation channel attention in every block\n"
            "- Global average pooling + dense classification head\n"
            "- Batch normalization & dropout throughout for regularization"
        )
    else:
        st.markdown(
            f"**{config.MODEL_ARCHITECTURE.title()}** backbone, ImageNet-pretrained, fine-tuned "
            "in two phases:\n\n"
            "1. **Head warm-up** — backbone frozen, train a new dense classification head\n"
            "2. **Fine-tuning** — unfreeze the top backbone layers at a low learning rate\n\n"
            "Trained with class-weighted binary cross-entropy, label smoothing, mixed precision, "
            "early stopping on validation AUC, and learning-rate plateau reduction."
        )

ui.render_divider()
st.info(
    "**Disclaimer:** PulmoVision AI is an educational / portfolio project demonstrating an "
    "end-to-end deep learning pipeline. It is **not** a certified medical device and must never "
    "be used for real clinical diagnosis. Always consult a qualified radiologist or physician.",
    icon="🏥",
)

ui.render_footer()
