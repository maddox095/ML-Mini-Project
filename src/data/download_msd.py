"""Register or explicitly download the MSD 10k archive without mutating it."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from src.data.common import append_manifest, load_config, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="Existing MSD archive to register")
    parser.add_argument("--download", action="store_true", help="Download the configured archive")
    parser.add_argument("--url", help="Override the configured archive URL")
    args = parser.parse_args()
    if bool(args.archive) == bool(args.download):
        parser.error("supply exactly one of --archive or --download")

    config = load_config()
    url = args.url or config["sources"]["msd_subset_url"]
    if args.download:
        destination = repo_path("data/raw/msd/msd_subset.tar.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing raw archive: {destination}")
        print(f"Downloading to {destination}; this archive is large and may take a while.")
        with urllib.request.urlopen(url) as response, destination.open("xb") as output:
            shutil.copyfileobj(response, output)
        archive = destination
    else:
        archive = repo_path(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    append_manifest(archive, source="Million Song Dataset 10k subset", source_url=url)
    print(f"Registered immutable raw archive: {archive}")


if __name__ == "__main__":
    main()
