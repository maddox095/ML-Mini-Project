"""Export the frozen v1 decision tree for the static browser demonstration.

The browser uses this JSON representation only for inference.  The joblib
pipeline remains the authoritative archived model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/best_pipeline.joblib"
OUTPUT_PATH = ROOT / "site/dist/model.json"
FEATURES = ["tempo", "loudness", "duration", "artist_score"]


def main() -> None:
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline.named_steps["model"]
    tree = model.tree_
    nodes = []
    for index in range(tree.node_count):
        counts = tree.value[index][0]
        total = float(counts.sum())
        nodes.append(
            {
                "feature": int(tree.feature[index]),
                "threshold": float(tree.threshold[index]),
                "left": int(tree.children_left[index]),
                "right": int(tree.children_right[index]),
                "hit_probability": float(counts[1] / total),
            }
        )
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "model": "decision_tree",
                "version": "final_suite_v1",
                "features": FEATURES,
                "threshold": 0.5,
                "nodes": nodes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
