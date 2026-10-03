"""Load the verified final pipeline and predict using the four real inputs."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.data.common import repo_path, sha256
from src.models.evaluate import prediction_scores


def predict(tempo, loudness, duration, artist_score):
    values = np.array([tempo, loudness, duration, artist_score], dtype=float)
    if not np.isfinite(values).all() or tempo <= 0 or duration <= 0 or artist_score not in (0, 1):
        raise ValueError("Inputs must be finite; tempo/duration positive and Artist Score 0 or 1")
    metadata = json.loads(repo_path("models/metadata.json").read_text(encoding="utf-8"))
    path = repo_path("models/best_pipeline.joblib")
    if sha256(path) != metadata["sha256"]:
        raise ValueError("Saved final model checksum differs from metadata")
    pipeline = joblib.load(path)
    X = pd.DataFrame([dict(zip(["tempo", "loudness", "duration", "artist_score"], values))])
    scores, kind = prediction_scores(pipeline, X)
    return {"predicted_hit": int(pipeline.predict(X)[0]), kind: float(scores[0]),
            "model": metadata["selection"]["model"], "version": metadata["final_run_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tempo", type=float, required=True)
    parser.add_argument("--loudness", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--artist-score", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.tempo, args.loudness, args.duration, args.artist_score), indent=2))
