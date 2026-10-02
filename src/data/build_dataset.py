"""Build the auditable, leakage-safe User 1 handoff table (Mode B by default)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.data.common import load_config, repo_path
from src.features.normalize_names import canonicalize_frame


@dataclass(frozen=True)
class BuildResult:
    model_table: pd.DataFrame
    match_audit: pd.DataFrame
    artist_score_audit: pd.DataFrame
    quality: pd.DataFrame
    feature_availability: pd.DataFrame


def _unique_tracks(tracks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically retain a single MSD row for each canonical identity."""
    stable = tracks.sort_values(["canonical_key", "year", "track_id"], na_position="last").copy()
    duplicates = stable[stable.duplicated("canonical_key", keep=False)].copy()
    return stable.drop_duplicates("canonical_key", keep="first").copy(), duplicates


def _first_billboard_events(entries: pd.DataFrame) -> pd.DataFrame:
    ordered = entries.sort_values(["canonical_key", "chart_date", "rank"], na_position="last")
    first = ordered.drop_duplicates("canonical_key", keep="first").copy()
    return first.rename(columns={"chart_date": "first_chart_date"})


def _fuzzy_overlap_audit(candidates: pd.DataFrame, hit_events: pd.DataFrame, title_threshold: float, artist_threshold: float) -> pd.DataFrame:
    """Surface only strong fuzzy candidate overlaps; never silently relabel them."""
    hit_titles = hit_events["title_key"].tolist()
    by_title = hit_events.set_index("title_key", drop=False)
    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        match = process.extractOne(candidate.title_key, hit_titles, scorer=fuzz.ratio, score_cutoff=title_threshold)
        if not match:
            continue
        hit = by_title.iloc[match[2]]
        artist_score = fuzz.ratio(candidate.artist_key, hit.artist_key)
        if artist_score >= artist_threshold:
            rows.append({"track_id": candidate.track_id, "artist_name": candidate.artist_name, "title": candidate.title, "artist_key": candidate.artist_key, "title_key": candidate.title_key, "match_method": "fuzzy_overlap_candidate", "match_score": round((match[1] + artist_score) / 2, 2), "matched_artist": hit.artist_name, "matched_title": hit.title, "duplicate_status": "review_or_remove", "resolution": "not_used_as_negative"})
    return pd.DataFrame(rows)


def _artist_scores(table: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign artist_score using strictly earlier Billboard events, with evidence."""
    events = history.sort_values(["artist_key", "first_chart_date", "title"]).copy()
    by_artist = {key: group for key, group in events.groupby("artist_key", sort=False)}
    scored, evidence, scores = table.copy(), [], []
    for row in scored.itertuples(index=False):
        row_date = pd.Timestamp(row.reference_date)
        prior = by_artist.get(row.artist_key, pd.DataFrame())
        if not prior.empty:
            prior = prior.loc[prior["first_chart_date"] < row_date]
        if prior.empty:
            scores.append(0)
            evidence.append({"canonical_key": row.canonical_key, "artist_name": row.artist_name, "row_date": row_date, "artist_score": 0, "prior_hit_title": None, "prior_hit_date": None, "chronology_valid": True})
        else:
            hit = prior.iloc[-1]
            scores.append(1)
            evidence.append({"canonical_key": row.canonical_key, "artist_name": row.artist_name, "row_date": row_date, "artist_score": 1, "prior_hit_title": hit.title, "prior_hit_date": hit.first_chart_date, "chronology_valid": bool(hit.first_chart_date < row_date)})
    scored["artist_score"] = scores
    return scored, pd.DataFrame(evidence)


def _quality_table(tracks: pd.DataFrame, candidates: pd.DataFrame, positives: pd.DataFrame, negatives: pd.DataFrame, final: pd.DataFrame, duplicates: pd.DataFrame) -> pd.DataFrame:
    rows = [("msd_input_rows", len(tracks)), ("msd_in_window_candidates", len(candidates)), ("canonical_msd_duplicate_rows", len(duplicates)), ("positive_rows_feature_matched", len(positives)), ("negative_rows_after_overlap_removal", len(negatives)), ("final_rows", len(final)), ("final_hit_rows", int(final.hit.sum())), ("final_non_hit_rows", int((final.hit == 0).sum())), ("conflicting_label_identities", int(final.groupby("canonical_key").hit.nunique().gt(1).sum()))]
    quality = pd.DataFrame(rows, columns=["metric", "value"])
    for name, count in final.isna().sum().items():
        quality.loc[len(quality)] = [f"missing::{name}", int(count)]
    for year, count in final["reference_date"].dt.year.value_counts().sort_index().items():
        quality.loc[len(quality)] = [f"year::{year}", int(count)]
    return quality


def build_handoff(msd_tracks: pd.DataFrame, billboard_entries: pd.DataFrame, config: dict) -> BuildResult:
    """Create the model-ready interim table and every User 1 audit artifact."""
    strip_versions = config["matching"].get("strip_title_versions", False)
    tracks = canonicalize_frame(msd_tracks, strip_versions=strip_versions)
    entries = canonicalize_frame(billboard_entries, strip_versions=strip_versions)
    entries["chart_date"] = pd.to_datetime(entries["chart_date"], errors="coerce")
    entries = entries.dropna(subset=["chart_date"])
    start, end = pd.Timestamp(config["date_window"]["start"]), pd.Timestamp(config["date_window"]["end"])
    tracks["year"] = pd.to_numeric(tracks["year"], errors="coerce")
    candidates = tracks.loc[tracks.year.between(start.year, end.year)].copy()
    candidates, duplicates = _unique_tracks(candidates)
    hit_events = _first_billboard_events(entries.loc[entries.chart_date.between(start, end)].copy())

    # Positives must join to MSD to use only the stated, current-compliant Mode-B features.
    positive = hit_events.merge(candidates, on="canonical_key", how="inner", suffixes=("_billboard", "_msd"))
    positive = positive.rename(columns={"artist_name_msd": "artist_name", "title_msd": "title"})
    positive["artist_key"] = positive["artist_key_msd"]
    positive["title_key"] = positive["title_key_msd"]
    positive["reference_date"], positive["hit"], positive["source"] = positive["first_chart_date"], 1, "billboard_matched_to_msd"
    positive["match_method"], positive["match_score"] = "exact", 100.0

    exact_overlap = candidates.merge(hit_events[["canonical_key"]], on="canonical_key", how="inner")
    negative = candidates.loc[~candidates.canonical_key.isin(hit_events.canonical_key)].copy()
    fuzzy = pd.DataFrame()
    if config["matching"].get("fuzzy_enabled", True):
        fuzzy = _fuzzy_overlap_audit(negative, hit_events, config["matching"]["fuzzy_title_threshold"], config["matching"]["fuzzy_artist_threshold"])
        if not fuzzy.empty:
            negative = negative.loc[~negative.track_id.isin(fuzzy.track_id)].copy()
    # MSD has only a year. Jan 1 is conservative: prior history must predate that year.
    negative["reference_date"] = pd.to_datetime(negative.year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    negative["hit"], negative["source"] = 0, "msd_non_hit_candidate"
    negative["match_method"], negative["match_score"] = "no_billboard_overlap", np.nan
    if config["balancing"].get("enabled", True):
        desired = int(round(len(positive) * float(config["balancing"].get("negative_to_positive_ratio", 1.0))))
        negative = negative.sample(n=min(desired, len(negative)), random_state=config["seed"]).copy()
    keep = ["track_id", "song_id", "title", "artist_name", "artist_key", "title_key", "canonical_key", "year", "duration", "tempo", "loudness", "reference_date", "hit", "source", "match_method", "match_score"]
    final = pd.concat([positive.reindex(columns=keep), negative.reindex(columns=keep)], ignore_index=True)
    final, artist_audit = _artist_scores(final, hit_events)
    final = final.sort_values(["reference_date", "canonical_key", "hit"]).reset_index(drop=True)
    if final.empty:
        raise ValueError("No rows produced: check the date window and MSD/Billboard identity coverage")
    if final.groupby("canonical_key").hit.nunique().max() > 1:
        raise AssertionError("Conflicting labels remain for a canonical identity")
    if not set(final.hit.unique()).issubset({0, 1}):
        raise AssertionError("Target contains values other than 0/1")
    if not artist_audit["chronology_valid"].all():
        raise AssertionError("Artist Score audit contains future evidence")

    audit_columns = ["track_id", "artist_name", "title", "artist_key", "title_key", "match_method", "match_score", "duplicate_status", "resolution"]
    exact_audit = exact_overlap.assign(match_method="exact_overlap", match_score=100.0, duplicate_status="billboard_hit", resolution="removed_from_negative_pool").reindex(columns=audit_columns)
    positive_audit = positive.assign(duplicate_status="positive_feature_matched", resolution="included_as_positive").reindex(columns=audit_columns)
    duplicate_audit = duplicates.assign(match_method="duplicate_msd_identity", match_score=np.nan, duplicate_status="duplicate", resolution="deterministic_first_row_retained").reindex(columns=audit_columns)
    audit = pd.concat([exact_audit, positive_audit, duplicate_audit, fuzzy], ignore_index=True, sort=False)
    features = config.get("mode_b_features", ["tempo", "loudness", "duration"])
    availability = pd.DataFrame({"feature": features, "source": "Million Song Dataset", "missing_rows": [int(final[f].isna().sum()) for f in features], "model_input": True})
    quality = _quality_table(tracks, candidates, positive, negative, final, duplicates)
    return BuildResult(final, audit, artist_audit, quality, availability)


def write_eda(result: BuildResult, path: Path, config: dict) -> None:
    table, numeric = result.model_table, config.get("mode_b_features", [])
    lines = ["# User 1 EDA summary", "", f"- Reproduction mode: **{config['reproduction_mode']}** (MSD-native methodology reproduction).", f"- Rows: {len(table):,}; hits: {int(table.hit.sum()):,}; non-hits: {int((table.hit == 0).sum()):,}.", "- Artist Score uses a strictly earlier Billboard event; see `artist_score_audit.csv`.", "- Billboard outcome metadata is excluded from the model feature list.", "", "## Feature availability", "", result.feature_availability.to_markdown(index=False), "", "## Numeric descriptive statistics", "", table.groupby("hit")[numeric].describe().transpose().to_markdown(), "", "## Correlation notes", "", "Correlations are descriptive only. No Billboard rank, peak, or weeks-on-chart field is a feature.", "", table[numeric + ["artist_score", "hit"]].corr(numeric_only=True).round(3).to_markdown(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msd", default="data/interim/msd_tracks.parquet")
    parser.add_argument("--billboard", default="data/interim/billboard_entries.parquet")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    result = build_handoff(pd.read_parquet(repo_path(args.msd)), pd.read_parquet(repo_path(args.billboard)), config)
    interim, reports = repo_path("data/interim"), repo_path("reports")
    interim.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    result.model_table.to_parquet(interim / "model_table.parquet", index=False)
    result.match_audit.to_csv(reports / "match_audit.csv", index=False)
    result.artist_score_audit.to_csv(reports / "artist_score_audit.csv", index=False)
    result.quality.to_csv(reports / "data_quality.csv", index=False)
    result.feature_availability.to_csv(reports / "feature_availability.csv", index=False)
    write_eda(result, reports / "eda_summary.md", config)
    print(f"Wrote User 1 handoff: {len(result.model_table):,} rows")


if __name__ == "__main__":
    main()
