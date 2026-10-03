"""Regression checks for shared replay boundaries and generated entry points."""

import runpy
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import source_files
from src.models.bundles import write_replay_scripts
from src.models.validation import make_folds, replay_partition


def grouped_data():
    X = pd.DataFrame({"tempo": np.arange(60)}, index=np.arange(100, 160))
    y = pd.Series([0, 1] * 30, index=X.index)
    metadata = pd.DataFrame({"artist_key": [f"artist{i//2}" for i in range(60)]}, index=X.index)
    config = {"seed": 42, "cv": {"outer_folds": 3, "inner_folds": 2}}
    return X, y, metadata, config


@pytest.mark.parametrize("stage,inner_fold", [("outer_fold_2", 1), ("full_training_tuning", 2)])
def test_replay_uses_exact_recorded_fold_indices(stage, inner_fold):
    X, y, metadata, config = grouped_data()
    fitting = np.arange(len(y))
    seed = config["seed"]
    if stage.startswith("outer"):
        outer = make_folds(y, metadata, "artist_disjoint_partition", count=3, seed=seed)
        fitting, _ = outer[1]
        seed += 2
    inner = make_folds(y.iloc[fitting], metadata.iloc[fitting], "artist_disjoint_partition", count=2, seed=seed)
    training, validation = inner[inner_fold - 1]
    fitting_X, fitting_y, validation_X, validation_y = replay_partition(
        X, y, metadata, protocol="artist_disjoint_partition", stage=stage, config=config, inner_fold=inner_fold)
    pd.testing.assert_frame_equal(fitting_X, X.iloc[fitting[training]])
    pd.testing.assert_frame_equal(validation_X, X.iloc[fitting[validation]])
    assert fitting_X.index.equals(fitting_y.index)
    assert validation_X.index.equals(validation_y.index)
    assert set(metadata.loc[fitting_X.index].artist_key).isdisjoint(metadata.loc[validation_X.index].artist_key)


@pytest.mark.parametrize("stage,inner_fold", [
    ("outer_fold_0", None), ("outer_fold_4", None), ("unknown", None),
    ("full_training_tuning", 0), ("full_training_tuning", 3),
])
def test_replay_rejects_invalid_stage_or_fold(stage, inner_fold):
    X, y, metadata, config = grouped_data()
    with pytest.raises(ValueError, match="stage|fold"):
        replay_partition(X, y, metadata, protocol="artist_disjoint_partition", stage=stage,
                         config=config, inner_fold=inner_fold)


def test_generated_setting_scripts_never_fit_when_imported(tmp_path, monkeypatch):
    module = ModuleType("fake_replay")
    calls = []
    module.replay = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "fake_replay", module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    settings = tmp_path / "hyperparameters"
    settings.mkdir()
    write_replay_scripts(settings, "setting_0000", module="fake_replay", function="replay")
    for suffix in ["train", "test"]:
        script = settings / f"setting_0000_{suffix}.py"
        runpy.run_path(str(script))
    assert not calls
    runpy.run_path(str(settings / "setting_0000_test.py"), run_name="__main__")
    assert len(calls) == 1 and calls[0][1] == {"inner_fold": 1}


def test_snapshots_include_refactored_dependencies_and_test_configuration():
    names = source_files()
    assert {"src/models/validation.py", "src/models/artifacts.py", "src/models/bundles.py", "pytest.ini"} <= set(names)
    assert not any("__pycache__" in name for name in names)
