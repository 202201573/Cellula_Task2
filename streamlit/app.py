# app.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from transformers import pipeline
from imagecaption import generate_caption

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Toxic Content Classifier + Image Captioning", layout="wide")
st.title("Task 1: Toxic Content Classification (Text + Image Captioning)")

# ----------------------------
# Default paths
# ----------------------------
DEFAULT_DB_PATH = "database.csv"      # auto-updating log file (required)
DEFAULT_DATASET_PATH = "dataset.csv"  # your dataset (view only)

# ----------------------------
# Database schema (required)
# ----------------------------
DB_COLUMNS = [
    "timestamp",
    "input_type",      # "text" or "image"
    "content",         # user text OR generated caption
    "model_name",
    "label",           # SAFE / TOXIC (after threshold mapping)
    "score",           # toxicity probability (0..1)
]

def load_db(path: str) -> pd.DataFrame:
    """Load DB CSV. If missing, create empty DB with required columns."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=DB_COLUMNS)
    else:
        df = pd.DataFrame(columns=DB_COLUMNS)

    for c in DB_COLUMNS:
        if c not in df.columns:
            df[c] = None

    return df[DB_COLUMNS]

def append_to_db(path: str, row: Dict[str, Any]) -> None:
    """Append a row to the DB CSV and save immediately (auto-update)."""
    df = load_db(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

# ----------------------------
# Text classifier
# ----------------------------
@st.cache_resource
def load_text_classifier(model_id: str):
    """Load Hugging Face text classification pipeline."""
    return pipeline("text-classification", model=model_id, tokenizer=model_id)

def classify_text(clf, text: str, threshold: float) -> Tuple[str, float]:
    """
    Many toxicity models return label 'toxic' with a score.
    We convert score to SAFE/TOXIC using a threshold.
    """
    out = clf(text)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        score = float(out[0].get("score", 0.0))
        final_label = "TOXIC" if score >= threshold else "SAFE"
        return final_label, score
    return "UNKNOWN", 0.0

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

db_path = st.sidebar.text_input("Database CSV path (auto-updates)", value=DEFAULT_DB_PATH)
dataset_path = st.sidebar.text_input("Dataset CSV path (view only)", value=DEFAULT_DATASET_PATH)

st.sidebar.subheader("Text Classification Model")
tox_model_id = st.sidebar.text_input("Model ID (Hugging Face)", value="unitary/toxic-bert")

# ✅ Threshold slider (this fixes your issue)
tox_threshold = st.sidebar.slider("Toxicity threshold", 0.0, 1.0, 0.50, 0.01)
st.sidebar.caption("If score ≥ threshold → TOXIC, else SAFE.")

st.sidebar.subheader("Image Caption Model (BLIP)")
caption_model_id = st.sidebar.text_input("BLIP model ID", value="Salesforce/blip-image-captioning-base")

st.sidebar.caption("The DB CSV updates whenever you submit text or an image caption.")

clf = load_text_classifier(tox_model_id)

# ----------------------------
# Tabs
# ----------------------------
tab_text, tab_image, tab_db, tab_dataset = st.tabs(
    ["📝 Text input", "🖼️ Image input", "📄 Database", "📊 Dataset"]
)

# ----------------------------
# Text tab
# ----------------------------
with tab_text:
    st.subheader("Text Toxicity Classification")

    user_text = st.text_area("Enter text:", height=150, placeholder="Type text to classify...")

    if st.button("Classify Text", type="primary"):
        if not user_text.strip():
            st.error("Please enter text first.")
        else:
            label, score = classify_text(clf, user_text, threshold=tox_threshold)
            st.success(f"Result: **{label}** (toxicity score={score:.4f})")

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_type": "text",
                "content": user_text,
                "model_name": tox_model_id,
                "label": label,
                "score": score,
            }
            append_to_db(db_path, row)
            st.info(f"Saved to DB: `{db_path}` (auto-updated)")

# ----------------------------
# Image tab
# ----------------------------
with tab_image:
    st.subheader("Image → Caption (BLIP) → Toxicity Classification")

    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width="stretch")

        if st.button("Generate Caption + Classify", type="primary"):
            with st.spinner("Generating caption using BLIP..."):
                caption = generate_caption(img, model_name=caption_model_id)

            st.write("**Generated caption:**")
            st.code(caption)

            with st.spinner("Classifying caption toxicity..."):
                label, score = classify_text(clf, caption, threshold=tox_threshold)

            st.success(f"Result: **{label}** (toxicity score={score:.4f})")

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_type": "image",
                "content": caption,
                "model_name": tox_model_id,
                "label": label,
                "score": score,
            }
            append_to_db(db_path, row)
            st.info(f"Saved to DB: `{db_path}` (auto-updated)")

# ----------------------------
# Database tab
# ----------------------------
with tab_db:
    st.subheader("Database Viewer (Auto-updated CSV log)")

    df_db = load_db(db_path)
    st.write(f"Rows: **{len(df_db)}**")
    st.dataframe(df_db, width="stretch")

    st.download_button(
        label="Download database CSV",
        data=df_db.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(db_path),
        mime="text/csv",
    )

# ----------------------------
# Dataset tab (read-only)
# ----------------------------
with tab_dataset:
    st.subheader("Dataset Viewer (Read-only)")

    if os.path.exists(dataset_path):
        df_data = pd.read_csv(dataset_path)
        st.write(f"Rows: **{len(df_data)}**")
        st.write(f"Columns: {list(df_data.columns)}")
        st.dataframe(df_data.head(200), width="stretch")
    else:
        st.warning(f"Dataset not found: {dataset_path}")
