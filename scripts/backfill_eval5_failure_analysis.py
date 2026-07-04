#!/usr/bin/env python
"""Backfill enhanced failure analysis for completed eval5 runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import FAILURE_CATEGORY_LABELS, build_failure_case_analysis


METHODS = (
    "qwen35_08b_baseline",
    "agentdog15_08b_reference",
    "agentdog15_fg_08b_reference",
)
DATASET_FILES = {
    "atbench": "summer_camp_ATBench300.json",
    "rjudge": "summer_camp_rjudge.json",
}
PER_ROUND_HEADER = [
    "method",
    "dataset",
    "round",
    "accuracy",
    "macro_f1",
    "unsafe_f1",
    "avg_output_tokens",
    "invalid_output_rate",
    "inference_seconds",
    "avg_latency_seconds_per_sample",
    "actual_eval_batch_size",
]
for _failure_key in FAILURE_CATEGORY_LABELS:
    PER_ROUND_HEADER.extend(
        [f"failure_{_failure_key}_count", f"failure_{_failure_key}_rate"]
    )
AGG_HEADER = [
    "method",
    "dataset",
    "num_rounds",
    "mean_accuracy",
    "std_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_inference_seconds",
    "mean_avg_output_tokens",
    "mean_invalid_output_rate",
]
for _failure_key in FAILURE_CATEGORY_LABELS:
    AGG_HEADER.append(f"mean_failure_{_failure_key}_rate")


def log(message: str) -> None:
    print(f"李文博_{message}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(header: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = []
        for key in header:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["dataset"]), []).append(row)

    aggregates = []
    for (method, dataset), group in sorted(grouped.items()):
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in group]

        acc = values("accuracy")
        macro = values("macro_f1")
        aggregate = {
            "method": method,
            "dataset": dataset,
            "num_rounds": len(group),
            "mean_accuracy": statistics.fmean(acc),
            "std_accuracy": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "mean_macro_f1": statistics.fmean(macro),
            "std_macro_f1": statistics.stdev(macro) if len(macro) > 1 else 0.0,
            "mean_inference_seconds": statistics.fmean(values("inference_seconds")),
            "mean_avg_output_tokens": statistics.fmean(values("avg_output_tokens")),
            "mean_invalid_output_rate": statistics.fmean(values("invalid_output_rate")),
        }
        for key in FAILURE_CATEGORY_LABELS:
            aggregate[f"mean_failure_{key}_rate"] = statistics.fmean(
                float(row.get(f"failure_{key}_rate", 0.0)) for row in group
            )
        aggregates.append(aggregate)
    return aggregates


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    aggregates = aggregate_rows(rows)
    write_csv(output_root / "summary_eval5.csv", rows, PER_ROUND_HEADER)
    write_csv(output_root / "summary_eval5_aggregate.csv", aggregates, AGG_HEADER)
    (output_root / "summary_eval5.md").write_text(
        "# Per-round metrics\n\n"
        + markdown_table(PER_ROUND_HEADER, rows)
        + "\n# Aggregate metrics\n\n"
        + markdown_table(AGG_HEADER, aggregates),
        encoding="utf-8",
    )
    (output_root / "summary_eval5.json").write_text(
        json.dumps(
            {"per_round": rows, "aggregate": aggregates},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_tensorboard_backfill(
    log_dir: str | None,
    dataset: str,
    metrics: dict[str, Any],
    failure_analysis: str,
) -> None:
    if not log_dir:
        return
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_threads.isdigit() or int(omp_threads) < 1:
        os.environ["OMP_NUM_THREADS"] = "8"
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=log_dir)
    try:
        for key in FAILURE_CATEGORY_LABELS:
            count_key = f"failure_{key}_count"
            rate_key = f"failure_{key}_rate"
            writer.add_scalar(f"{dataset}/{count_key}", metrics.get(count_key, 0), 1)
            writer.add_scalar(f"{dataset}/{rate_key}", metrics.get(rate_key, 0.0), 1)
        writer.add_text(f"{dataset}/failure_case_analysis", failure_analysis, 1)
        writer.add_text(f"{dataset}/failure_case_analysis_enhanced", failure_analysis, 1)
    finally:
        writer.flush()
        writer.close()


def backfill_round(
    output_root: Path,
    method: str,
    round_idx: int,
    source_data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    round_dir = output_root / method / f"round_{round_idx:02d}"
    summary_path = round_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for dataset, source_rows in source_data.items():
        prediction_path = round_dir / f"predictions_{dataset}.jsonl"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing predictions: {prediction_path}")
        predictions = read_jsonl(prediction_path)
        failure_analysis, failure_stats = build_failure_case_analysis(
            predictions=predictions,
            source_rows=source_rows,
            dataset_name=dataset,
        )
        metrics = summary["datasets"][dataset]
        for key, count in failure_stats.items():
            metrics[f"failure_{key}_count"] = count
            metrics[f"failure_{key}_rate"] = count / len(predictions) if predictions else 0.0
        metrics["failure_case_analysis"] = failure_analysis
        (round_dir / f"failure_analysis_{dataset}.md").write_text(
            failure_analysis + "\n",
            encoding="utf-8",
        )
        write_tensorboard_backfill(
            log_dir=summary.get("tensorboard_log_dir"),
            dataset=dataset,
            metrics=metrics,
            failure_analysis=failure_analysis,
        )
        row = {
            "method": method,
            "dataset": dataset,
            "round": round_idx,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "unsafe_f1": metrics["unsafe_f1"],
            "avg_output_tokens": metrics["avg_output_tokens"],
            "invalid_output_rate": metrics["invalid_output_rate"],
            "inference_seconds": metrics["inference_seconds"],
            "avg_latency_seconds_per_sample": metrics["avg_latency_seconds_per_sample"],
            "actual_eval_batch_size": summary["eval_batch_size_actual"],
        }
        for key in FAILURE_CATEGORY_LABELS:
            row[f"failure_{key}_count"] = metrics.get(f"failure_{key}_count", 0)
            row[f"failure_{key}_rate"] = metrics.get(f"failure_{key}_rate", 0.0)
        rows.append(row)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/eval5")
    parser.add_argument(
        "--test-data-dir",
        default="/root/autodl-tmp/agentdog_lite_summer_camp/data/2026_summer_camp_teseset",
    )
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    data_dir = Path(args.test_data_dir)
    source_data = {
        dataset: json.loads((data_dir / filename).read_text(encoding="utf-8"))
        for dataset, filename in DATASET_FILES.items()
    }

    per_round_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for round_idx in range(1, args.rounds + 1):
            per_round_rows.extend(backfill_round(output_root, method, round_idx, source_data))
            log(f"[backfill] updated method={method} round={round_idx:02d}")

    write_outputs(output_root, per_round_rows)
    log(f"[backfill] wrote enhanced eval5 summaries under {output_root}")


if __name__ == "__main__":
    main()
