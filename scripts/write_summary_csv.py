#!/usr/bin/env python
"""Aggregate per-method summary.json files into outputs/summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


HEADER = [
    "method",
    "dataset",
    "num_samples",
    "accuracy",
    "unsafe_precision",
    "unsafe_recall",
    "unsafe_f1",
    "macro_f1",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "invalid_output_rate",
    "strict_json_rate",
    "over_refusal_rate_safe_to_unsafe",
    "miss_rate_unsafe_to_safe",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_methods.yaml")
    parser.add_argument("--output", default="outputs/summary.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for method, spec in config["methods"].items():
        summary_path = Path(spec["output_dir"]) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary for {method}: {summary_path}")
        summary = load_json(summary_path)
        for dataset, metrics in summary["datasets"].items():
            rows.append({"method": method, "dataset": dataset, **metrics})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[summary] wrote {output}")


if __name__ == "__main__":
    main()

