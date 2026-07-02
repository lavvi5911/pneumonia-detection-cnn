"""
app_helpers.py
==============
Shared helpers imported by app.py and every page in pages/. Keeps caching,
styling, and small UI-component builders in one place so pages stay thin.
"""

from __future__ import annotations

import streamlit as st
import tensorflow as tf

import config
from src import model_builder  # noqa: F401  (import registers custom Keras layers used by saved models)


def inject_global_css():
    css_path = config.ROOT_DIR / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def page_setup(page_title: str, page_icon: str = "🫁"):
    st.set_page_config(
        page_title=f"{page_title} · {config.APP_TITLE}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()


@st.cache_resource(show_spinner="Loading model weights…")
def load_model(model_path: str | None = None) -> tf.keras.Model:
    path = model_path or str(config.BEST_MODEL_PATH)
    return tf.keras.models.load_model(path)


def model_available() -> bool:
    return config.BEST_MODEL_PATH.exists() or config.FINAL_MODEL_PATH.exists()


def resolve_model_path() -> str:
    if config.BEST_MODEL_PATH.exists():
        return str(config.BEST_MODEL_PATH)
    if config.FINAL_MODEL_PATH.exists():
        return str(config.FINAL_MODEL_PATH)
    raise FileNotFoundError(
        "No trained model found in models/. Run `python -m src.train` first, "
        "or drop a best_model.keras file into the models/ folder."
    )


def render_hero(title: str, tagline: str, badge_text: str = "Model online"):
    st.markdown(
        f"""
        <div class="pv-hero">
          <span class="pv-eyebrow"><span class="pv-pulse-dot"></span> {badge_text}</span>
          <h1>{title}</h1>
          <p class="pv-tagline">{tagline}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body_html: str, icon: str = ""):
    st.markdown(
        f"""
        <div class="pv-card">
          <h4>{icon}&nbsp; {title}</h4>
          <p>{body_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step(number: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="pv-step">
          <div class="pv-step-num">{number}</div>
          <div>
            <strong>{title}</strong><br/>
            <span style="color:var(--muted); font-size:0.88rem;">{description}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_divider():
    st.markdown('<hr class="pv-divider" />', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        f"""
        <div class="pv-footer">
          {config.APP_TITLE} · Educational demo, not a certified diagnostic device ·
          Built with TensorFlow &amp; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_class(risk_level: str) -> str:
    return risk_level.lower().replace(" ", "-")
