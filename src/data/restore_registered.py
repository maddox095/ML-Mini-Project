"""Restore public source files only when bytes match the registered provenance."""

import argparse
import json
import os
import time
import urllib.request

import pandas as pd

from src.data.common import repo_path, sha256


def restore(source="all", max_seconds=300):
    if max_seconds <= 0:
        raise ValueError("Use a positive download time budget")
    manifest = pd.read_csv(repo_path("data/raw/MANIFEST.csv"))
    records = []
    for row in manifest.itertuples(index=False):
        kind = "billboard" if "billboard/" in row.artifact else "msd"
        if source != "all" and source != kind:
            continue
        target = repo_path(row.artifact)
        record = {"source": kind, "path": row.artifact, "expected_sha256": row.sha256,
                  "expected_bytes": int(row.bytes)}
        if target.exists():
            record["status"] = "verified existing" if sha256(target) == row.sha256 and target.stat().st_size == row.bytes else "existing file does not match; untouched"
            records.append(record)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".restore_download")
        if temporary.exists():
            raise FileExistsError(f"Preserve an earlier unfinished download: {temporary}")
        urls = [row.source_url]
        if row.source_url.startswith("http://"):
            urls.append(row.source_url.replace("http://", "https://", 1))
        failures = []
        for url in urls:
            try:
                deadline = time.monotonic() + max_seconds
                print(f"Restoring {kind} from {url}", flush=True)
                with urllib.request.urlopen(url, timeout=30) as response, temporary.open("xb") as output:
                    while chunk := response.read1(64 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError("Download exceeded the time budget; unverified bytes discarded")
                        output.write(chunk)
                digest = sha256(temporary)
                if digest != row.sha256 or temporary.stat().st_size != row.bytes:
                    raise ValueError(f"Downloaded source changed: sha256={digest}, bytes={temporary.stat().st_size}")
                os.link(temporary, target)
                record.update(status="restored and verified", retrieved_url=url)
                break
            except Exception as error:
                failures.append(str(error))
            finally:
                temporary.unlink(missing_ok=True)
        else:
            record.update(status="unavailable; original provenance preserved", errors=failures)
        records.append(record)
        print(record["status"], flush=True)
    output = repo_path(f"reports/source_restore_v2_{source}.json")
    output.write_text(json.dumps({"sources": records}, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", "billboard", "msd"], default="all")
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    restore(args.source, args.max_seconds)
