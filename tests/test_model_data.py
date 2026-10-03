import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.build_features import make_pipeline, make_preprocessor
from src.models.data import load_partition


def test_preprocessing_learns_training_only_and_passes_binary_unchanged():
    train = pd.DataFrame({"tempo": [100.0, 120.0, np.nan], "loudness": [-10.0, -8.0, -6.0],
                          "duration": [180.0, 220.0, 260.0], "artist_score": [0, 1, 0],
                          "hit": [0, 1, 0], "peak_position": [100, 1, 90]})
    validation = train.iloc[[0]].copy()
    validation["tempo"] = 10000.0
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(train)
    means = preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_.copy()
    preprocessor.transform(validation)
    np.testing.assert_array_equal(means, preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_)
    assert means[0] == 110.0
    assert transformed.shape == (3, 4)
    np.testing.assert_array_equal(transformed[:, -1], train.artist_score)
    pipeline = make_pipeline(LogisticRegression(random_state=42))
    pipeline.fit(train, train.hit)
    assert pipeline.predict_proba(validation).shape == (1, 2)


@pytest.mark.parametrize("protocol", ["stratified_partition", "artist_disjoint_partition"])
def test_current_partitions_load_declared_features_only(protocol):
    X, y, metadata = load_partition(protocol, "train")
    assert list(X.columns) == ["tempo", "loudness", "duration", "artist_score"]
    assert set(y.unique()) == {0, 1}
    assert X.index.equals(y.index)
    assert metadata[protocol].eq("train").all()
    with pytest.raises(ValueError, match="reserved"):
        load_partition(protocol, "test")
    _, _, test = load_partition(protocol, "test", allow_test=True)
    assert set(metadata.track_id).isdisjoint(test.track_id)
    if protocol == "artist_disjoint_partition":
        assert set(metadata.artist_key).isdisjoint(test.artist_key)


def test_unknown_protocol_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        load_partition("P1_source_75_25", "train")
