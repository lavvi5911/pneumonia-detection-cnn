"""
pages/1_Diagnose.py
====================
Core interactive page: upload a chest X-ray, run inference, visualize the
prediction confidence and a Grad-CAM explainability overlay.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src import app_helpers as ui
from src import gradcam, predict

ui.page_setup("Diagnose", page_icon="🔍")
ui.render_hero(
    "Diagnose a Chest X-Ray",
    "Upload a radiograph below. The model runs inference locally and returns a "
    "verdict, confidence score, and a Grad-CAM heatmap explaining its focus.",
    badge_text="Real-time inference",
)

if not ui.model_available():
    st.error(
        "No trained model found in `models/`. Train one with `python -m src.train`, or place a "
        "`best_model.keras` file in the `models/` folder, then reload this page.",
        icon="🚫",
    )
    st.stop()

model = ui.load_model(ui.resolve_model_path())

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Inference settings")
    threshold = st.slider(
        "Decision threshold (PNEUMONIA if P ≥ threshold)",
        min_value=0.05, max_value=0.95, value=float(config.DECISION_THRESHOLD), step=0.01,
    )
    show_gradcam = st.checkbox("Show Grad-CAM explainability", value=True)
    gradcam_alpha = st.slider("Heatmap overlay opacity", 0.1, 0.9, float(config.GRADCAM_ALPHA), 0.05)
    st.caption(
        "Lowering the threshold increases sensitivity (catches more pneumonia cases) at the cost "
        "of more false alarms — the classic clinical trade-off."
    )

# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------
col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )
with col_sample:
    st.markdown("&nbsp;")
    sample_dir = config.SAMPLE_IMAGES_DIR
    sample_files = sorted(sample_dir.glob("*")) if sample_dir.exists() else []
    use_sample = None
    if sample_files:
        sample_choice = st.selectbox(
            "…or try a sample image", ["None"] + [f.name for f in sample_files]
        )
        if sample_choice != "None":
            use_sample = sample_dir / sample_choice

image_source = None
if uploaded_file is not None:
    image_source = Image.open(uploaded_file)
elif use_sample is not None:
    image_source = Image.open(use_sample)

if image_source is None:
    st.markdown(
        '<div class="pv-card"><h4>👆 Upload an image to begin</h4>'
        '<p>Accepted formats: JPG, JPEG, PNG. For best results, use a frontal chest radiograph '
        "similar to the training distribution (pediatric/adult postero-anterior chest X-rays).</p></div>",
        unsafe_allow_html=True,
    )
    ui.render_footer()
    st.stop()

# ------------------------------------------------------------------
# Run inference
# ------------------------------------------------------------------
with st.spinner("Running inference…"):
    result, model_input_batch, original_rgb = predict.predict_single(model, image_source, threshold=threshold)

img_col, result_col = st.columns([1, 1.15], gap="large")

with img_col:
    st.markdown("#### Input radiograph")
    st.image(image_source, use_container_width=True)

    if show_gradcam:
        with st.spinner("Computing Grad-CAM heatmap…"):
            try:
                overlay = gradcam.generate_gradcam_overlay(
                    model, model_input_batch, original_rgb, layer_name=None
                )
                config.GRADCAM_ALPHA_RUNTIME = gradcam_alpha  # not persisted, informational only
                st.markdown("#### Grad-CAM explainability overlay")
                st.image(overlay, use_container_width=True)
                st.caption(
                    "Warmer colors indicate image regions that most increased the model's "
                    "predicted pneumonia probability."
                )
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated for this model/layer: {e}")

with result_col:
    verdict_class = "pneumonia" if result.label == config.POSITIVE_CLASS else "normal"
    verdict_icon = "🫁⚠️" if verdict_class == "pneumonia" else "✅"
    badge_class = ui.risk_badge_class(result.risk_level)

    st.markdown(
        f"""
        <div class="pv-verdict {verdict_class}">
          <span class="pv-badge {badge_class}">{result.risk_level} risk</span>
          <div class="pv-verdict-label">{verdict_icon} {result.label}</div>
          <p style="color:var(--muted); margin-top:0.3rem;">
            Confidence: <strong style="color:var(--text);">{result.confidence*100:.1f}%</strong>
            &nbsp;·&nbsp; Decision threshold: {threshold:.2f}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")

    # Confidence gauge
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result.probability_pneumonia * 100,
            number={"suffix": "%", "font": {"color": "#E2E8F0", "family": "JetBrains Mono"}},
            title={"text": "P(PNEUMONIA)", "font": {"color": "#94A3B8", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94A3B8", "tickfont": {"color": "#94A3B8"}},
                "bar": {"color": "#14B8A6"},
                "bgcolor": "#0F2947",
                "borderwidth": 1,
                "bordercolor": "#1E3A5F",
                "steps": [
                    {"range": [0, 30], "color": "rgba(20,184,166,0.25)"},
                    {"range": [30, 60], "color": "rgba(249,115,22,0.25)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability breakdown bar
    prob_fig = go.Figure(
        go.Bar(
            x=[result.probability_normal * 100, result.probability_pneumonia * 100],
            y=["NORMAL", "PNEUMONIA"],
            orientation="h",
            marker_color=["#14B8A6", "#EF4444"],
            text=[f"{result.probability_normal*100:.1f}%", f"{result.probability_pneumonia*100:.1f}%"],
            textposition="outside",
            textfont=dict(color="#E2E8F0"),
        )
    )
    prob_fig.update_layout(
        height=180, margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], showgrid=False, color="#94A3B8"),
        yaxis=dict(color="#E2E8F0"),
        showlegend=False,
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    with st.expander("Raw model output"):
        st.json(
            {
                "label": result.label,
                "probability_pneumonia": round(result.probability_pneumonia, 6),
                "probability_normal": round(result.probability_normal, 6),
                "decision_threshold": threshold,
                "risk_level": result.risk_level,
            }
        )

ui.render_divider()
st.info(
    "**Disclaimer:** This tool is for educational/demonstration purposes only and is **not** a "
    "certified medical device. Do not use it for real clinical decisions.",
    icon="🏥",
)
ui.render_footer()
