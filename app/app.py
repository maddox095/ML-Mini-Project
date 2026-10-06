"""Streamlit browser demo for the selected HitPredict random forest pipeline.

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
    """Load the published pipeline and verify its current metadata checksum."""
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
st.caption("A four-input research reproduction demo using the selected random forest")
st.info(
    "This is an experimental research model. Its probability describes a balanced "
    "research sample and is not a real-world commercial-success probability."
)

try:
    pipeline, metadata = load_model()
except Exception as error:
    st.error(f"Unable to load the published model: {error}")
    st.stop()

st.caption(f"Model: Random forest · Version: {metadata['final_run_id']}")

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
    st.caption("Fixed classifier threshold: 0.50 (exact ties predict non-hit). Your inputs are processed only in this app session.")

with st.expander("Published model evidence"):
    metrics = metadata["test_metrics"]
    st.write(
        f"This forest's recorded evaluation uses {metadata['test_rows']} artist-disjoint held-out songs. "
        "It had the highest observed accuracy in the original comparison and was chosen "
        "for deployment after those results were reviewed. This is not a fresh independent test."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Accuracy": metrics["accuracy"], "Precision": metrics["precision"], "Recall": metrics["recall"], "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"]}]
        ).style.format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1": "{:.2%}", "ROC-AUC": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    matrix = metadata["confusion_matrix"]
    st.write(f"Held-out confusion matrix: TN {matrix['tn']} · FP {matrix['fp']} · "
             f"FN {matrix['fn']} · TP {matrix['tp']}.")
    st.dataframe(pd.DataFrame([[matrix['tn'], matrix['fp']], [matrix['fn'], matrix['tp']]],
        index=["Actual non-hit", "Actual hit"], columns=["Predicted non-hit", "Predicted hit"]),
        width="stretch")
    st.caption(f"Model version: {metadata['final_run_id']} · Features: {', '.join(metadata['features'])}")

with st.expander("Interpretation and limitations"):
    st.markdown(
        "- Artist Score uses only chart events strictly earlier than the song's reference date.\n"
        "- The model uses Million Song Dataset descriptors and historical Billboard labels.\n"
        "- Exact matching and finite chart history can leave label noise; a non-hit candidate is not proof that a song never charted."
    )
