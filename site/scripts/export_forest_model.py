"""Export the verified deployed forest and preprocessing for browser inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]


def export_model():
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
    path = ROOT / "models/best_pipeline.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Active forest checksum does not match deployment metadata")
    pipeline = joblib.load(path)
    model = pipeline.named_steps["model"]
    if not isinstance(model, RandomForestClassifier) or list(model.classes_) != [0, 1]:
        raise ValueError("The browser exporter requires the deployed binary random forest")
    numeric = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
    if "scaler" in numeric.named_steps:
        raise ValueError("This forest exporter expects unscaled numeric inputs")
    expected = ["numeric__tempo", "numeric__loudness", "numeric__duration", "binary__artist_score"]
    if list(pipeline.named_steps["preprocessor"].get_feature_names_out()) != expected:
        raise ValueError("Unexpected transformed input order")
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        nodes = []
        for index in range(tree.node_count):
            values = tree.value[index][0]
            nodes.append({"feature": int(tree.feature[index]),
                          "threshold": float(tree.threshold[index]),
                          "left": int(tree.children_left[index]),
                          "right": int(tree.children_right[index]),
                          "hit_probability": float(values[1] / values.sum())})
        trees.append(nodes)
    return {"model": "random_forest", "version": metadata["final_run_id"],
            "features": metadata["features"], "threshold": metadata["threshold"],
            "positive_class_rule": "probability > threshold; ties predict non-hit",
            "input_dtype": "float32", "tree_count": len(trees),
            "imputer_medians": [float(value) for value in numeric.named_steps["imputer"].statistics_],
            "model_sha256": metadata["sha256"], "test_metrics": metadata["test_metrics"],
            "confusion_matrix": metadata["confusion_matrix"], "trees": trees}


def main():
    destination = ROOT / "site/dist/model.json"
    destination.write_text(json.dumps(export_model(), separators=(",", ":"), allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
