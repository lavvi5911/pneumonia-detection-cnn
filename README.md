# 🫁 PulmoVision AI — Pneumonia Detection from Chest X-Rays

An end-to-end, production-style deep learning project: a custom + transfer-learning CNN that
classifies chest X-rays as **NORMAL** or **PNEUMONIA**, with full training/evaluation tooling,
Grad-CAM explainability, and a polished multi-page **Streamlit** app ready for one-click
deployment on **Streamlit Community Cloud**.

> Built as an extended, "production-grade" version of a course CNN lecture — this project adds
> residual + attention architectures, two-phase transfer learning, full evaluation tooling,
> Grad-CAM explainability, tests, CI, and a deployable front end on top of the lecture's basics.

---

## ✨ Features

- **Two model families**: a from-scratch residual CNN with Squeeze-and-Excitation attention
  (`PulmoNet`), and transfer learning on EfficientNetB0 / DenseNet121 / ResNet50V2.
- **Two-phase transfer learning**: frozen-backbone head warm-up, then selective fine-tuning of
  the top backbone layers at a low learning rate.
- **Robust training pipeline**: `tf.data` input pipeline with domain-appropriate augmentation,
  class-weighted loss for imbalance, label smoothing, mixed precision, early stopping, LR
  scheduling, checkpointing, TensorBoard + CSV logging.
- **Full evaluation suite**: accuracy, ROC-AUC, PR-AUC, sensitivity/specificity, confusion
  matrix, Youden's-J optimal threshold, full `sklearn` classification report — all exported as
  JSON + styled plots consumed directly by the app.
- **Grad-CAM explainability**: a from-scratch gradient-based class activation map implementation
  that works on both the custom CNN and nested transfer-learning backbones.
- **Professional Streamlit front end**: custom design system (dark clinical theme, glassmorphism
  cards, animated hero scan-line, confidence gauges, risk badges), 4 pages (Diagnose, Model
  Insights, How It Works, About), adjustable decision threshold, sample images for reviewers.
- **Deployment-ready**: `requirements.txt` pinned, `.streamlit/config.toml` themed, GitHub
  Actions CI smoke test, `pytest` unit tests, `.gitignore` tuned for ML repos.

---

## 🗂️ Project structure

```
pneumonia-detection-cnn/
├── app.py                      # Streamlit home page (entry point)
├── config.py                   # single source of truth for paths & hyperparameters
├── requirements.txt
├── .streamlit/config.toml      # Streamlit theme
├── .github/workflows/ci.yml    # import + architecture smoke test on push
├── pages/                      # Streamlit multipage app
│   ├── 1_🔍_Diagnose.py         # upload → predict → Grad-CAM
│   ├── 2_📊_Model_Insights.py   # training curves, ROC/PR, confusion matrix
│   ├── 3_🧠_How_It_Works.py     # architecture & pipeline deep-dive
│   └── 4_ℹ️_About.py            # dataset, ethics, tech stack
├── src/
│   ├── data_loader.py           # tf.data pipeline, augmentation, class weights
│   ├── model_builder.py         # custom CNN + transfer-learning architectures
│   ├── train.py                 # two-phase training entry point
│   ├── evaluate.py               # metrics + plots on the test set
│   ├── gradcam.py                # Grad-CAM implementation
│   ├── predict.py                # single-image inference utility
│   ├── app_helpers.py            # Streamlit caching + UI component helpers
│   └── utils.py                  # seeding, JSON I/O, threshold search
├── scripts/
│   ├── download_dataset.py       # pulls the Kaggle dataset via kagglehub
│   └── prepare_sample_images.py  # copies a few demo images into sample_images/
├── tests/test_smoke.py           # architecture/shape sanity tests (no dataset needed)
├── assets/style.css              # design system for the Streamlit app
├── sample_images/                # a few demo X-rays for the app's "try a sample" picker
├── models/                       # trained .keras files land here (gitignored)
└── artifacts/                    # metrics.json, history.json, evaluation plots (gitignored)
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

You said you already have the [Kaggle chest X-ray pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
locally with `train/`, `test/`, and `val/` folders. Either:

**Option A — point config at your existing folder** (no copying needed):
```bash
export PNEUMONIA_DATA_DIR="/path/to/your/chest_xray"   # Windows (PowerShell): $env:PNEUMONIA_DATA_DIR="C:\path\to\chest_xray"
```

**Option B — copy it into the repo's `data/` folder:**
```bash
mkdir -p data
cp -r /path/to/your/chest_xray data/chest_xray
```

**Option C — download fresh via the Kaggle API:**
```bash
pip install kagglehub
python scripts/download_dataset.py
```

Either way, `config.py` expects this structure:
```
chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}
```

> ⚠️ The Kaggle dataset's `val/` folder only has 16 images. For more reliable validation
> metrics, `src/data_loader.py` works fine as-is, but consider re-splitting `train/` (e.g. an
> 90/10 split) if you want a larger validation set — the pipeline doesn't require the original
> split sizes.

### 3. Train a model

```bash
# Best accuracy, ImageNet-pretrained (recommended default):
python -m src.train --architecture efficientnet

# Or the from-scratch custom CNN:
python -m src.train --architecture custom_cnn

# Other transfer-learning options:
python -m src.train --architecture densenet
python -m src.train --architecture resnet
```

This writes `models/best_model.keras`, `models/final_model.keras`, and
`artifacts/history.json`. Training on a GPU (Colab, local CUDA, etc.) is strongly recommended —
EfficientNetB0 fine-tuning on ~5,200 images takes a few minutes per epoch on CPU vs. seconds on GPU.

### 4. Evaluate

```bash
python -m src.evaluate
```

Generates `artifacts/metrics.json` plus `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`,
and `training_curves.png` — all consumed by the app's **Model Insights** page.

### 5. (Optional) Prepare demo images

```bash
python scripts/prepare_sample_images.py --n_per_class 3
```

### 6. Run the app locally

```bash
streamlit run app.py
```

Open the printed local URL — you'll land on the home page, with **Diagnose**, **Model Insights**,
**How It Works**, and **About** in the sidebar.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (see below for what to commit vs. gitignore).
2. Since trained `.keras` model files are large and are gitignored by default, you need the
   model available to the deployed app. Pick one:
   - **Git LFS** — `git lfs track "*.keras"` and commit `models/best_model.keras` through LFS.
   - **External hosting** — upload `best_model.keras` to a Hugging Face Hub model repo, a GitHub
     Release asset, or cloud storage, and have `app.py`/`app_helpers.py` download it at startup
     (check for the file, `requests.get(...)` it into `models/` if missing, then load).
   - **Small model** — if you trained `custom_cnn`, it may be small enough (a few tens of MB) to
     commit directly without LFS.
3. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account, pick this
   repo, set the main file path to `app.py`, and deploy.
4. Streamlit Cloud installs from `requirements.txt` and picks up `.streamlit/config.toml`
   automatically — no extra configuration needed.
5. The **Diagnose** and **Model Insights** pages gracefully degrade with informative warnings if
   `models/` or `artifacts/` are empty, so the app never crashes on a fresh deploy — it just tells
   the visitor what's missing.

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

The smoke tests build every architecture on random tensors and check output shapes/ranges — they
don't need the dataset or a trained model, so they're safe to run in CI (see
`.github/workflows/ci.yml`).

---

## 🏗️ Architecture summary

| | Custom CNN (`PulmoNet`) | Transfer learning |
|---|---|---|
| Backbone | From scratch: stem + 4 residual stages (64→128→256→512 filters) + SE attention | EfficientNetB0 / DenseNet121 / ResNet50V2, ImageNet weights |
| Params | ~11M | ~5–8M (backbone) + head |
| Training | Single phase | Two phases: head warm-up → fine-tune |
| Best for | Understanding CNN internals, no external weights needed | Best raw accuracy, faster convergence |

Full write-up with diagrams-in-prose lives on the app's **How It Works** page.

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is **not** a certified
medical device and must **never** be used for real clinical diagnosis or treatment decisions.
Always consult a qualified radiologist or physician. See the app's **About** page for a full
discussion of dataset bias and limitations.

---

## 📚 Acknowledgements

- Dataset: Kermany, D.S. et al., *"Identifying Medical Diagnoses and Treatable Diseases by
  Image-Based Deep Learning,"* Cell, 2018 — mirrored on
  [Kaggle by Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- Course reference: *Data Science & AI Masters — From Python to Gen AI* (Udemy), CNN lecture.
- Built with TensorFlow, scikit-learn, OpenCV, Streamlit, and Plotly.
