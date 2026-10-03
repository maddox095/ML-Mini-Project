"""Regression checks for current-schema verification and stale-artifact loading."""

import shutil

import pandas as pd
import pytest

from src.data import common
from src.data.verify_handoff import verify_handoff
from src.models.data import load_partition


@pytest.fixture
def isolated_repository(tmp_path, monkeypatch):
    for name in ["configs/data.yaml", "data/interim/model_table.parquet",
                 "data/raw/MANIFEST.csv", "reports/artist_score_audit.csv",
                 "reports/feature_availability.csv", "reports/data_quality.csv",
                 "data/processed/user2_splits.csv", "data/processed/user2_split_manifest.json"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.repo_path(name), destination)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    return tmp_path


def test_current_handoff_is_verified_without_claiming_source_acceptance(isolated_repository):
    result = verify_handoff()
    assert result["integrity_passed"]
    assert not result["raw_inputs_verified"]
    assert not result["accepted_for_final_modeling"]
    assert all(result["checks"].values())


@pytest.mark.parametrize("corruption,check", [
    ("duplicate", "one_row_per_identity"),
    ("nonfinite", "finite_features"),
    ("future_evidence", "positive_artist_scores_have_prior_evidence"),
    ("missing_column", "required_columns"),
])
def test_corrupted_handoff_fails_integrity(isolated_repository, corruption, check):
    source = isolated_repository / "data/interim/model_table.parquet"
    table = pd.read_parquet(source)
    if corruption == "future_evidence":
        path = isolated_repository / "reports/artist_score_audit.csv"
        audit = pd.read_csv(path)
        audit.loc[audit.artist_score.eq(1).idxmax(), "prior_hit_date"] = "2099-01-01"
        audit.to_csv(path, index=False)
    else:
        if corruption == "duplicate":
            table.loc[1, "canonical_key"] = table.loc[0, "canonical_key"]
        elif corruption == "nonfinite":
            table.loc[0, "tempo"] = float("inf")
        else:
            table = table.drop(columns="duration")
        table.to_parquet(source, index=False)
    result = verify_handoff()
    assert not result["integrity_passed"]
    assert not result["checks"][check]


@pytest.mark.parametrize("name,message", [
    ("data/interim/model_table.parquet", "Model table has changed"),
    ("data/processed/user2_splits.csv", "Split assignments have changed"),
])
def test_loader_rejects_changed_artifacts(isolated_repository, name, message):
    with (isolated_repository / name).open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValueError, match=message):
        load_partition("stratified_partition", "train")
