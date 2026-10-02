"""Build the User 1 Mode B handoff from full MSD summary data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.common import load_config, repo_path
from src.data.load_billboard import parse_billboard
from src.features.normalize_names import canonicalize_frame


def _unique(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["canonical_key", "year", "track_id"], na_position="last")
    duplicates = ordered[ordered.duplicated("canonical_key", keep=False)].copy()
    return ordered.drop_duplicates("canonical_key", keep="first").copy(), duplicates


def _load_billboard_history(raw_path: Path, history_start: str, history_end: str) -> pd.DataFrame:
    with raw_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_billboard(payload, history_start, history_end)


def _prior_artist_score(table: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.sort_values(["artist_key", "chart_date", "title"])
    by_artist = {key: group for key, group in history.groupby("artist_key", sort=False)}
    scores, audit = [], []
    for row in table.itertuples(index=False):
        prior = by_artist.get(row.artist_key, pd.DataFrame())
        if not prior.empty:
            prior = prior.loc[prior.chart_date < row.reference_date]
        if prior.empty:
            scores.append(0)
            audit.append({"canonical_key": row.canonical_key, "artist_name": row.artist_name, "row_date": row.reference_date, "artist_score": 0, "prior_hit_title": None, "prior_hit_date": None, "chronology_valid": True})
        else:
            evidence = prior.iloc[-1]
            scores.append(1)
            audit.append({"canonical_key": row.canonical_key, "artist_name": row.artist_name, "row_date": row.reference_date, "artist_score": 1, "prior_hit_title": evidence.title, "prior_hit_date": evidence.chart_date, "chronology_valid": bool(evidence.chart_date < row.reference_date)})
    result = table.copy()
    result["artist_score"] = scores
    return result, pd.DataFrame(audit)


def build_dataset(msd: pd.DataFrame, billboard_history: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = config["label_window"]
    start, end = pd.Timestamp(labels["start"]), pd.Timestamp(labels["end"])
    msd = canonicalize_frame(msd, strip_versions=config["matching"]["strip_title_versions"])
    msd["year"] = pd.to_numeric(msd.year, errors="coerce")
    candidates = msd.loc[msd.year.between(start.year, end.year)].copy()
    candidates, duplicates = _unique(candidates)
    history = canonicalize_frame(billboard_history, strip_versions=config["matching"]["strip_title_versions"])
    history["chart_date"] = pd.to_datetime(history.chart_date, errors="coerce")
    history = history.dropna(subset=["chart_date"])
    labels_df = history.loc[history.chart_date.between(start, end)].sort_values(["canonical_key", "chart_date", "rank"])
    hits = labels_df.drop_duplicates("canonical_key", keep="first").rename(columns={"chart_date": "first_chart_date"})

    positives = hits.merge(candidates, on="canonical_key", how="inner", suffixes=("_billboard", "_msd"))
    positives["artist_name"] = positives["artist_name_msd"]
    positives["title"] = positives["title_msd"]
    positives["artist_key"] = positives["artist_key_msd"]
    positives["title_key"] = positives["title_key_msd"]
    positives["reference_date"] = positives["first_chart_date"]
    positives["hit"] = 1
    positives["source"] = "billboard_matched_to_msd_summary"
    positives["match_method"] = "exact"
    positives["match_score"] = 100.0

    unmatched = hits.loc[~hits.canonical_key.isin(candidates.canonical_key)].copy()
    unmatched["match_method"] = "unresolved_exact"
    unmatched["match_score"] = np.nan
    unmatched["resolution"] = "not_included_without_feature_match"
    exact_overlap = candidates.loc[candidates.canonical_key.isin(hits.canonical_key)].copy()
    negatives = candidates.loc[~candidates.canonical_key.isin(hits.canonical_key)].copy()
    negatives["reference_date"] = pd.to_datetime(negatives.year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    negatives["hit"] = 0
    negatives["source"] = "msd_summary_non_hit_candidate"
    negatives["match_method"] = "no_exact_billboard_overlap"
    negatives["match_score"] = np.nan

    negative_pool_size = len(negatives)
    target = int(config["balancing"]["target_rows_per_class"])
    n = min(target, len(positives), len(negatives))
    positives = positives.sample(n=n, random_state=config["seed"])
    negatives = negatives.sample(n=n, random_state=config["seed"])
    # The official summary file exposes ``danceability`` and ``energy``, but
    # those fields are zero for this release.  They are therefore deliberately
    # excluded from the modelling handoff rather than presented as features.
    columns = ["track_id", "song_id", "title", "artist_name", "artist_key", "title_key", "canonical_key", "year", "duration", "tempo", "loudness", "reference_date", "hit", "source", "match_method", "match_score"]
    final = pd.concat([positives.reindex(columns=columns), negatives.reindex(columns=columns)], ignore_index=True)
    final, artist_audit = _prior_artist_score(final, history)
    if final.groupby("canonical_key").hit.nunique().gt(1).any():
        raise AssertionError("Conflicting labels remain after overlap removal")
    if not artist_audit.chronology_valid.all():
        raise AssertionError("Artist Score contains future information")
    audit = pd.concat(
        [
            positives.assign(resolution="included_as_positive").reindex(columns=["canonical_key", "artist_name", "title", "match_method", "match_score", "resolution"]),
            exact_overlap.assign(match_method="exact_overlap", match_score=100.0, resolution="removed_from_negative_pool").reindex(columns=["canonical_key", "artist_name", "title", "match_method", "match_score", "resolution"]),
            unmatched.reindex(columns=["canonical_key", "artist_name", "title", "match_method", "match_score", "resolution"]),
            duplicates.assign(match_method="duplicate_msd_identity", match_score=np.nan, resolution="deterministic_first_row_retained").reindex(columns=["canonical_key", "artist_name", "title", "match_method", "match_score", "resolution"]),
        ], ignore_index=True
    )
    quality = pd.DataFrame(
        [
            ("msd_summary_rows", len(msd)), ("msd_candidates_in_label_window", len(candidates)),
            ("unique_billboard_hits_in_label_window", len(hits)), ("positive_feature_matches_before_sampling", len(hits) - len(unmatched)),
            ("unresolved_billboard_hits", len(unmatched)), ("negative_pool_before_balancing", negative_pool_size),
            ("final_rows", len(final)), ("final_hits", int(final.hit.sum())), ("final_non_hits", int((final.hit == 0).sum())),
            ("conflicting_label_identities", int(final.groupby("canonical_key").hit.nunique().gt(1).sum())),
        ], columns=["metric", "value"]
    )
    return final.sort_values(["reference_date", "canonical_key"]).reset_index(drop=True), audit, artist_audit, quality


def write_eda(table: pd.DataFrame, report_path: Path, features: list[str]) -> None:
    """Write the initial, non-model EDA required for the User 1 handoff."""
    summary = table.groupby("hit")[features].describe().transpose().round(3).to_markdown()
    correlations = table[features + ["artist_score", "hit"]].corr(numeric_only=True).round(3).to_markdown()
    lines = [
        "# User 1 EDA summary", "",
        f"- Rows: {len(table):,}; hits: {int(table.hit.sum()):,}; non-hits: {int((table.hit == 0).sum()):,}.",
        "- Mode B uses MSD-derived features; this is a methodology reproduction, not an exact Spotify-feature reproduction.",
        "- `artist_score` uses Billboard events from 1986 onward that occur strictly before the song reference date.",
        "- Chart rank, peak position, and weeks on chart are excluded from the model table.", "",
        "## Numeric descriptive statistics by class", "", summary, "",
        "## Correlation notes", "",
        "Correlations are descriptive only and must not be interpreted as causal effects.", "", correlations, "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msd", default="data/interim/msd_summary_tracks.parquet")
    parser.add_argument("--billboard-raw", default="data/raw/billboard/all.json")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    history_window = config["artist_history_window"]
    history = _load_billboard_history(repo_path(args.billboard_raw), history_window["start"], history_window["end"])
    final, audit, artist_audit, quality = build_dataset(pd.read_parquet(repo_path(args.msd)), history, config)
    output = repo_path("data/interim")
    reports = repo_path("reports")
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output / "model_table.parquet", index=False)
    audit.to_csv(reports / "match_audit.csv", index=False)
    artist_audit.to_csv(reports / "artist_score_audit.csv", index=False)
    quality.to_csv(reports / "data_quality.csv", index=False)
    features = config["mode_b_features"] + ["artist_score"]
    pd.DataFrame(
        {
            "feature": features,
            "source": ["Million Song Dataset summary file"] * (len(features) - 1) + ["Prior Billboard history"],
            "missing_rows": [int(final[feature].isna().sum()) for feature in features],
            "model_input": True,
        }
    ).to_csv(reports / "feature_availability.csv", index=False)
    write_eda(final, reports / "eda_summary.md", features[:-1])
    print(f"Wrote User 1 handoff: {len(final):,} rows")


if __name__ == "__main__":
    main()
