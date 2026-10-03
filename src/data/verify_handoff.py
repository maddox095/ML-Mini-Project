"""Check the current handoff and report source availability separately."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.data.common import load_config, model_features, repo_path, sha256


def verify_handoff() -> dict:
    config = load_config()
    path = repo_path("data/interim/model_table.parquet")
    table = pd.read_parquet(path)
    features = model_features(config)
    required = ["track_id", "song_id", "title", "artist_name", "artist_key", "title_key",
                "canonical_key", "year", "reference_date", "hit", *features]
    checks = {"required_columns": set(required).issubset(table.columns)}
    result = {"input_sha256": sha256(path), "rows": len(table),
              "dataset_version": config["dataset_version"], "features": features}
    if not checks["required_columns"]:
        return {**result, "integrity_passed": False, "accepted_for_final_modeling": False,
                "checks": checks, "missing_columns": sorted(set(required) - set(table.columns))}
    dates = pd.to_datetime(table.reference_date, errors="coerce")
    numeric = table[features].apply(pd.to_numeric, errors="coerce")
    labels = config["label_window"]
    history = config["artist_history_window"]
    checks.update({
        "nonempty": not table.empty,
        "one_row_per_identity": table.canonical_key.is_unique,
        "unique_track_ids": table.track_id.is_unique,
        "no_conflicting_labels": not table.groupby("canonical_key").hit.nunique().gt(1).any(),
        "binary_target_both_classes": set(table.hit.unique()) == {0, 1},
        "balanced_classes": table.hit.eq(0).sum() == table.hit.eq(1).sum(),
        "binary_artist_score": table.artist_score.isin([0, 1]).all(),
        "required_fields_present": not table[required].isna().any().any(),
        "nonempty_identities": table[["track_id", "song_id", "artist_key", "title_key"]].apply(lambda s: s.str.strip().ne("")).all().all(),
        "finite_features": np.isfinite(numeric.to_numpy()).all(),
        "positive_tempo_and_duration": (numeric.tempo > 0).all() and (numeric.duration > 0).all(),
        "dates_in_label_window": dates.between(labels["start"], labels["end"]).all(),
        "history_contains_label_window": pd.Timestamp(history["start"]) <= pd.Timestamp(labels["start"]) <= pd.Timestamp(labels["end"]) <= pd.Timestamp(history["end"]),
        "model_features_exclude_outcomes": not bool(set(features) & {"hit", "rank", "peak_position", "weeks_on_chart", "source", "match_score"}),
        "handoff_excludes_matching_provenance": not bool(set(table.columns) & {"source", "match_method", "match_score"}),
    })
    audit = pd.read_csv(repo_path("reports/artist_score_audit.csv"))
    audit_columns = {"canonical_key", "artist_name", "row_date", "artist_score", "prior_hit_date", "prior_hit_title", "chronology_valid"}
    checks["artist_audit_schema"] = audit_columns.issubset(audit.columns)
    checks["artist_audit_covers_all_rows"] = ("canonical_key" in audit
        and audit.canonical_key.is_unique and len(audit) == len(table)
        and set(audit.canonical_key) == set(table.canonical_key))
    if checks["artist_audit_schema"] and checks["artist_audit_covers_all_rows"] and checks["one_row_per_identity"]:
        joined = table[["canonical_key", "artist_name", "reference_date", "artist_score"]].merge(
            audit, on="canonical_key", suffixes=("_table", "_audit"), validate="one_to_one")
        scored = joined.artist_score_table.eq(1)
        prior = pd.to_datetime(joined.prior_hit_date, errors="coerce")
        reference = pd.to_datetime(joined.reference_date, errors="coerce")
        checks.update({
            "artist_audit_scores_agree": joined.artist_score_table.eq(joined.artist_score_audit).all(),
            "artist_audit_names_agree": joined.artist_name_table.eq(joined.artist_name_audit).all(),
            "artist_audit_dates_agree": reference.eq(pd.to_datetime(joined.row_date, errors="coerce")).all(),
            "positive_artist_scores_have_prior_evidence": (prior[scored] < reference[scored]).all() and joined.loc[scored, "prior_hit_title"].fillna("").str.strip().ne("").all(),
            "prior_evidence_in_history_window": prior[scored].between(history["start"], history["end"]).all(),
            "zero_artist_scores_have_no_prior_evidence": joined.loc[~scored, ["prior_hit_date", "prior_hit_title"]].isna().all().all(),
            "chronology_flags_pass": audit.chronology_valid.eq(True).all(),
        })
    availability = pd.read_csv(repo_path("reports/feature_availability.csv"))
    checks["feature_sources_documented"] = set(features).issubset(availability.feature) and availability.source.notna().all()
    quality = pd.read_csv(repo_path("reports/data_quality.csv")).set_index("metric").value
    checks["quality_counts_agree"] = all(quality.get(key) == value for key, value in {
        "final_rows": len(table), "final_hits": int(table.hit.eq(1).sum()),
        "final_non_hits": int(table.hit.eq(0).sum())}.items())
    manifest = pd.read_csv(repo_path("data/raw/MANIFEST.csv"))
    raw = []
    for row in manifest.itertuples(index=False):
        artifact = repo_path(row.artifact)
        present = artifact.is_file()
        raw.append({"artifact": row.artifact, "present": present,
                    "recorded_sha256": row.sha256,
                    "checksum_verified": bool(present and sha256(artifact) == row.sha256),
                    "size_verified": bool(present and artifact.stat().st_size == row.bytes)})
    raw_verified = bool(raw) and all(item["checksum_verified"] and item["size_verified"] for item in raw)
    intermediates = {name: repo_path(name).is_file() for name in (
        "data/interim/msd_summary_tracks.parquet", "data/interim/billboard_entries.parquet")}
    limitations = [
        "Mode B uses three MSD audio descriptors plus Artist Score; it is not a numerical reproduction of the original Spotify-feature experiment.",
        "Exact matching and finite chart history can leave label noise; an unmatched song is only a non-hit candidate.",
        "Non-hit reference dates approximate release as January 1; dates are metadata, not model inputs.",
        "Saved Artist Score evidence validates chronology, not every source event or zero score without raw history.",
    ]
    if not raw_verified:
        limitations.insert(0, "Registered raw sources are unavailable or unverified locally; full-source extraction and labels cannot be independently reproduced in this checkout.")
    if not all(intermediates.values()):
        limitations.insert(1, "Extracted MSD/Billboard intermediate tables are absent locally.")
    integrity = bool(all(checks.values()))
    return {**result, "integrity_passed": integrity,
        "accepted_for_final_modeling": False,
        "status": "handoff integrity passed; source review required" if integrity else "handoff integrity failed",
        "checks": {key: bool(value) for key, value in checks.items()},
        "class_counts": {str(key): int(value) for key, value in table.hit.value_counts().items()},
        "distinct_artists": int(table.artist_key.nunique()),
        "reference_date_range": [str(dates.min()), str(dates.max())],
        "raw_inputs": raw, "raw_inputs_verified": raw_verified,
        "intermediate_tables_present": intermediates, "limitations": limitations,
        "review_policy": "Integrity checks do not independently certify source labels or establish final research acceptance.",
    }


def write_report(result: dict, path) -> None:
    checks = result["checks"]
    counts = result.get("class_counts", {})
    lines = ["# User 1 verification", "",
        f"- Dataset version: `{result['dataset_version']}`.",
        f"- Rows: {result['rows']:,}; hits: {counts.get('1', 0):,}; non-hits: {counts.get('0', 0):,}.",
        f"- Integrity checks: {sum(checks.values())}/{len(checks)} passed.",
        f"- Status: {result.get('status', 'handoff integrity failed')}.",
        f"- Dataset SHA-256: `{result['input_sha256']}`.", "",
        "This report is generated from the current table and audits. Regenerate it with",
        "`python -m src.data.verify_handoff` after data changes.", "",
        "## Checks", "", "| Check | Result |", "| --- | --- |"]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    lines.extend(["", "## Source availability", "", "| Artifact | Present | Hash and size verified |", "| --- | --- | --- |"])
    for item in result.get("raw_inputs", []):
        lines.append(f"| `{item['artifact']}` | {item['present']} | {item['checksum_verified'] and item['size_verified']} |")
    lines.extend(["", "## Remaining limitations", ""])
    lines.extend(f"- {limitation}" for limitation in result.get("limitations", []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/user1_verification.json")
    parser.add_argument("--markdown-output", default="reports/user1_verification.md")
    args = parser.parse_args()
    result = verify_handoff()
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, repo_path(args.markdown_output))
    print(f"Integrity: {'PASS' if result['integrity_passed'] else 'FAIL'}; {sum(result['checks'].values())}/{len(result['checks'])} checks passed")
    print(f"Saved {output}")
    if not result["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
