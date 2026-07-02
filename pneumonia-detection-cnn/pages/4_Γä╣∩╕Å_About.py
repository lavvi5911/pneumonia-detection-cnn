"""
pages/4_About.py
==================
Project context: dataset provenance, tech stack, limitations, and ethics.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src import app_helpers as ui

ui.page_setup("About", page_icon="ℹ️")
ui.render_hero(
    "About This Project",
    "Context on the dataset, the tech stack, and the limitations you should keep in mind.",
    badge_text="Project info",
)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("### 📁 Dataset")
    st.markdown(
        """
**Chest X-Ray Images (Pneumonia)** — Kaggle, curated by Paul Mooney
(originally from Kermany et al., *Cell*, 2018).

- 5,863 pediatric chest X-ray images (JPEG)
- 2 classes: `NORMAL`, `PNEUMONIA` (bacterial + viral pneumonia grouped together)
- Pre-split into `train/`, `val/`, and `test/` folders
- Source: [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset is **not bundled in this repository** (it's several hundred MB and Kaggle's terms
require downloading it directly) — see the README for setup instructions.
        """
    )

    st.markdown("### 🛠️ Tech stack")
    st.markdown(
        """
| Layer | Tools |
|---|---|
| Modeling | TensorFlow / Keras |
| Data pipeline | `tf.data`, OpenCV, Pillow |
| Evaluation | scikit-learn, Matplotlib, Seaborn |
| App / UI | Streamlit, Plotly |
| Explainability | Custom Grad-CAM implementation |
        """
    )

with c2:
    st.markdown("### ⚠️ Limitations & ethics")
    st.markdown(
        """
- **Not a medical device.** This project is for learning and portfolio purposes. It has not
  been validated, cleared, or approved for clinical use by any regulatory body.
- **Dataset bias.** The training data comes from a single pediatric hospital cohort in China (1–5
  year-old patients). Performance is unlikely to generalize to adult patients, different scanner
  hardware, or different patient populations without further validation.
- **No differential diagnosis.** The model outputs a binary NORMAL/PNEUMONIA score — it cannot
  distinguish pneumonia from other causes of lung opacity (e.g. edema, atelectasis, malignancy),
  nor bacterial from viral pneumonia.
- **False negatives carry real risk.** Any deployment in an assistive-triage context would need
  a carefully validated operating threshold, ideally tuned toward higher sensitivity, and should
  always keep a clinician in the loop.
- **Explainability ≠ correctness.** Grad-CAM shows *where* the model looked, not *why* it's
  correct — a heatmap over a plausible lung region is reassuring but not proof of a correct diagnosis.
        """
    )

    st.markdown("### 🔗 Links")
    st.markdown(
        """
- Course reference (lecture this project extends): *Data Science & AI Masters — From Python
  to Gen AI* (Udemy)
- Model architecture, training, and evaluation code: see `src/` in this repository
- Report a bug or suggest an improvement: open an issue on the project's GitHub repository
        """
    )

ui.render_divider()
st.markdown(
    f"""
<div class="pv-card">
  <h4>🙏 Acknowledgements</h4>
  <p>
    Dataset: Kermany, D.S. et al., "Identifying Medical Diagnoses and Treatable Diseases by
    Image-Based Deep Learning," <em>Cell</em>, 2018, mirrored on Kaggle by Paul Mooney.
    Pretrained backbones courtesy of the Keras Applications ImageNet weights.
    Built with TensorFlow and Streamlit as a demonstration of an end-to-end CNN project:
    data pipeline → model architecture → training → evaluation → explainability → deployment.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

ui.render_footer()
