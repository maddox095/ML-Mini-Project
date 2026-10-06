"""Keep CLI, Streamlit evidence and browser forest predictions consistent."""

import json
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data.common import sha256
from src.inference import predict

ROOT = Path(__file__).resolve().parents[1]


def test_active_forest_and_evidence_match_the_frozen_checkpoint():
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
    path = ROOT / "models/best_pipeline.joblib"
    assert metadata["selection"]["model"] == "random_forest"
    assert metadata["selection"]["selected_before_test"] is False
    assert sha256(path) == metadata["sha256"] == sha256(ROOT / metadata["selection"]["artifact"])
    assert isinstance(joblib.load(path).named_steps["model"], RandomForestClassifier)
    matrix = metadata["confusion_matrix"]
    assert matrix == {"tn": 465, "fp": 27, "fn": 144, "tp": 340}
    assert metadata["test_metrics"]["accuracy"] == (matrix["tn"] + matrix["tp"]) / sum(matrix.values())
    outcome = predict(120, -8, 210, 1)
    assert outcome["model"] == "random_forest"
    assert outcome["version"] == metadata["final_run_id"]


def test_browser_javascript_matches_pipeline_on_examples_and_float32_boundaries():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the actual browser scoring function")
    pipeline = joblib.load(ROOT / "models/best_pipeline.joblib")
    forest = pipeline.named_steps["model"]
    rng = np.random.default_rng(42)
    examples = np.column_stack([rng.uniform(30, 250, 20), rng.uniform(-40, -1, 20),
                               rng.uniform(60, 600, 20), rng.integers(0, 2, 20)]).tolist()
    for tree in forest.estimators_[:4]:
        for feature, threshold in zip(tree.tree_.feature[:8], tree.tree_.threshold[:8]):
            if 0 <= feature < 3:
                for offset in (-1e-7, 0, 1e-7):
                    row = [120., -8., 210., 1.]
                    row[feature] = threshold + offset
                    examples.append(row)
    script = """
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('site/dist/index.html','utf8');
const model = JSON.parse(fs.readFileSync('site/dist/model.json','utf8'));
const code = html.slice(html.indexOf('function score(values)'),html.indexOf("fetch('./model.json'"));
const context = {model, Math}; vm.createContext(context); vm.runInContext(code,context);
const rows = JSON.parse(process.argv[1]);
process.stdout.write(JSON.stringify(rows.map(row=>context.score(row))));
"""
    result = subprocess.run([node, "-e", script, json.dumps(examples)], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    browser = np.asarray(json.loads(result.stdout))
    frame = pd.DataFrame(examples, columns=["tempo", "loudness", "duration", "artist_score"])
    np.testing.assert_allclose(browser, pipeline.predict_proba(frame)[:, 1], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(browser > .5, pipeline.predict(frame))


def test_streamlit_shows_forest_evidence_and_predicts_two_examples():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app/app.py")).run(timeout=20)
    assert not app.exception
    assert any("random forest" in item.value.lower() for item in app.caption)
    evidence = app.dataframe[0].value
    assert evidence.iloc[0]["Accuracy"] == pytest.approx(805 / 976)
    assert any("TN 465" in item.value and "TP 340" in item.value for item in app.markdown)
    for tempo, loudness, duration, artist in [(120., -8., 210., 1), (90., -16., 180., 0)]:
        app.number_input[0].set_value(tempo)
        app.number_input[1].set_value(loudness)
        app.number_input[2].set_value(duration)
        app.radio[0].set_value(artist)
        app.button[0].click().run(timeout=20)
        assert not app.exception
        expected = predict(tempo, loudness, duration, artist)
        assert app.metric[0].value == f"{expected['probability']:.1%}"
        assert app.metric[1].value == ("Hit candidate" if expected["predicted_hit"] else "Non-hit candidate")
