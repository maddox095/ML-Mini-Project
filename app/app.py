"""Streamlit browser demo for the frozen HitPredict v1 pipeline.

Run locally from the repository root:
    streamlit run app/app.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
FEATURES = ["tempo", "loudness", "duration", "artist_score"]


@st.cache_resource
def load_model():
    """Load only the published, hash-verified v1 pipeline."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise RuntimeError("The saved model does not match its published metadata checksum.")
    return joblib.load(MODEL_PATH), metadata


def input_frame(tempo: float, loudness: float, duration: float, artist_score: int) -> pd.DataFrame:
    """Construct the named, ordered inputs consumed by the saved pipeline."""
    return pd.DataFrame([[tempo, loudness, duration, artist_score]], columns=FEATURES)


st.set_page_config(page_title="HitPredict demo", page_icon="♫", layout="centered")
st.title("HitPredict")
st.caption("A four-input research reproduction demo using the frozen v1 decision tree")
st.info(
    "This is an experimental research model. Its probability is calibrated to a balanced "
    "research sample and is not a real-world commercial-success probability."
)

try:
    pipeline, metadata = load_model()
except Exception as error:
    st.error(f"Unable to load the published model: {error}")
    st.stop()

with st.form("prediction"):
    tempo = st.number_input("Tempo (BPM)", min_value=1.0, max_value=300.0, value=120.0, step=0.1)
    loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=0.0, value=-8.0, step=0.1)
    duration = st.number_input("Duration (seconds)", min_value=1.0, max_value=3600.0, value=210.0, step=1.0)
    artist_score = st.radio(
        "Prior-hit Artist Score",
        options=[0, 1],
        format_func=lambda value: "1 — normalized artist had a prior chart hit" if value else "0 — no prior chart hit",
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    inputs = input_frame(tempo, loudness, duration, artist_score)
    probability = float(pipeline.predict_proba(inputs)[0, 1])
    predicted_hit = bool(pipeline.predict(inputs)[0])
    left, right = st.columns(2)
    left.metric("Predicted hit probability", f"{probability:.1%}")
    right.metric("Predicted class", "Hit candidate" if predicted_hit else "Non-hit candidate")
    st.caption("Fixed classifier threshold: 0.50. Your inputs are processed only in this app session.")

with st.expander("Published model evidence"):
    metrics = metadata["test_metrics"]
    st.write(
        "The selected model was evaluated once on 976 artist-disjoint held-out songs. "
        "The tree and its preprocessing were selected before this evaluation."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Accuracy": metrics["accuracy"], "Precision": metrics["precision"], "Recall": metrics["recall"], "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"]}]
        ).style.format("{:.1%}"),
        hide_index=True,
        width="stretch",
    )
    st.write("Held-out confusion matrix: TN 411 · FP 81 · FN 122 · TP 362.")
    st.caption(f"Model version: {metadata['final_run_id']} · Features: {', '.join(metadata['features'])}")

with st.expander("Interpretation and limitations"):
    st.markdown(
        "- Artist Score uses only chart events strictly earlier than the song's reference date.\n"
        "- The model uses Million Song Dataset descriptors and historical Billboard labels.\n"
        "- Exact matching and finite chart history can leave label noise; a non-hit candidate is not proof that a song never charted."
    )
