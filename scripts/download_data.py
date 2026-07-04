#!/usr/bin/env python
"""Download the allowed training data and held-out summer camp evaluation data."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


FILES = [
    (
        "AI45Research/AgentDoG1.0-Training-Data",
        "AgentDoG-BinarySafety/train.json",
        "data/raw/agentdog_training",
    ),
    (
        "AI45Research/AgentDoG1.0-Training-Data",
        "AgentDoG-FineGrainedTaxonomy/train.json",
        "data/raw/agentdog_training",
    ),
    (
        "AI45Research/2026_summer_camp_teseset",
        "summer_camp_ATBench300.json",
        "data/raw/summer_camp_teseset",
    ),
    (
        "AI45Research/2026_summer_camp_teseset",
        "summer_camp_rjudge.json",
        "data/raw/summer_camp_teseset",
    ),
]


def validate_json(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty file: {path}")
    with path.open(encoding="utf-8") as f:
        json.load(f)


def direct_download(repo_id: str, filename: str, output_path: Path) -> None:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        output_path.write_bytes(response.read())


def download_file(repo_id: str, filename: str, local_dir: str, retries: int) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=local_dir,
                )
            )
            validate_json(path)
            return path
        except Exception as exc:  # noqa: BLE001 - report full download failure path.
            last_error = exc
            print(f"[download] hub attempt {attempt}/{retries} failed for {filename}: {exc}")
            time.sleep(min(attempt * 2, 10))

    output_path = Path(local_dir) / filename
    try:
        print(f"[download] falling back to direct resolve URL for {filename}")
        direct_download(repo_id, filename, output_path)
        validate_json(output_path)
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download {repo_id}/{filename}") from (last_error or exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    for repo_id, filename, local_dir in FILES:
        print(f"[download] {repo_id}/{filename}")
        path = download_file(repo_id, filename, local_dir, args.retries)
        print(f"[download] ok: {path}")


if __name__ == "__main__":
    main()

