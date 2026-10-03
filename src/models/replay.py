"""Retrain a saved hyperparameter setting on its exact training/CV fold.

Run: python -m src.models.replay --setting hyperparameters/setting_0000.json
No final test partition is accessed. --inner-fold 1 reproduces that inner fit;
without it, fit all rows available at the recorded outer stage.
"""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.data.common import repo_path, sha256
from src.features.build_features import make_pipeline
from src.models.data import load_partition
from src.models.evaluate import prediction_scores, score_metrics
from src.models.registry import TREE_MODELS, make_estimator
from src.models.artifacts import save_pipeline, write_json
from src.models.validation import replay_partition


def run(setting_path, output, inner_fold=None):
    setting = json.loads(Path(setting_path).read_text(encoding="utf-8"))
    config, protocol = setting["config"], setting["protocol"]
    X, y, metadata = load_partition(protocol, "train")
    if sha256(repo_path("data/interim/model_table.parquet")) != setting["source_sha256"]:
        raise ValueError("Saved hyperparameter setting belongs to different data")
    if "splits_sha256" in setting and sha256(repo_path("data/processed/user2_splits.csv")) != setting["splits_sha256"]:
        raise ValueError("Saved hyperparameter setting belongs to different splits")
    X, y, validation_X, validation_y = replay_partition(
        X, y, metadata, protocol=protocol, stage=setting["stage"],
        config=config, inner_fold=inner_fold)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve earlier replay results; choose a new output")
    output.mkdir(parents=True)
    pipeline = make_pipeline(make_estimator(setting["model"], config),
        features=setting["features"], scale=setting["model"] not in TREE_MODELS)
    pipeline.set_params(**setting["parameters"])
    with threadpool_limits(limits=1):
        pipeline.fit(X, y)
    evidence = save_pipeline(pipeline, X.iloc[:32], output / "pipeline.joblib")
    result = {"setting": setting, "training_rows": len(y), "inner_fold": inner_fold,
              "estimator_parameters": pipeline.named_steps["model"].get_params(deep=False), **evidence}
    if validation_X is not None:
        scores, kind = prediction_scores(pipeline, validation_X)
        result["validation_metrics"] = score_metrics(validation_y, scores,
            predictions=pipeline.predict(validation_X), score_kind=kind)
    write_json(output / "metadata.json", result)
    print(f"Saved actual fitted pipeline and settings to {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--output", default="replayed_setting")
    parser.add_argument("--inner-fold", type=int)
    args = parser.parse_args()
    run(args.setting, args.output, args.inner_fold)
