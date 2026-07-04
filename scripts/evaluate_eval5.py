#!/usr/bin/env python
"""Run 3 checkpoints x 5 full batched eval rounds and aggregate metrics."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import (
    FAILURE_CATEGORY_LABELS,
    apply_test_data_dir,
    evaluate_method,
    resolve_eval_options,
)


METHODS = (
    "qwen35_08b_baseline",
    "agentdog15_08b_reference",
    "agentdog15_fg_08b_reference",
)
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
        aggregates.append(
            {
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
                **{
                    f"mean_failure_{key}_rate": statistics.fmean(
                        float(row.get(f"failure_{key}_rate", 0.0)) for row in group
                    )
                    for key in FAILURE_CATEGORY_LABELS
                },
            }
        )
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


def add_fg_method(config: dict[str, Any]) -> None:
    config.setdefault("methods", {})
    config["methods"]["agentdog15_fg_08b_reference"] = {
        "model_path": "models/AgentDoG1.5-FG-Qwen3.5-0.8B",
        "adapter_path": None,
        "output_dir": "outputs/reference_agentdog15_fg_08b",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_methods.yaml")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/eval5")
    parser.add_argument("--test-data-dir", default="/root/autodl-tmp/agentdog_lite_summer_camp/data/2026_summer_camp_teseset")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sort-by-length", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--auto-reduce-batch-on-oom",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    add_fg_method(config)
    apply_test_data_dir(config, args.test_data_dir)
    options = resolve_eval_options(config, args)
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []

    for method in METHODS:
        if method not in config["methods"]:
            raise KeyError(f"Missing eval method: {method}")
        model_path = Path(config["methods"][method]["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {method}: {model_path}")

    for method in METHODS:
        for round_idx in range(1, args.rounds + 1):
            round_name = f"round_{round_idx:02d}"
            round_output_dir = output_root / method / round_name
            run_config = copy.deepcopy(config)
            run_config["methods"][method]["output_dir"] = str(round_output_dir)
            run_config["methods"][method]["log_name"] = f"{method}_round{round_idx:02d}"
            log(f"[eval5] start method={method} {round_name}")
            summary = evaluate_method(
                config=run_config,
                method=method,
                limit=args.limit,
                options=options,
                output_dir_override=str(round_output_dir),
                log_name_override=f"{method}_round{round_idx:02d}",
            )
            for dataset, metrics in summary["datasets"].items():
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
                    "avg_latency_seconds_per_sample": metrics[
                        "avg_latency_seconds_per_sample"
                    ],
                    "actual_eval_batch_size": summary["eval_batch_size_actual"],
                }
                for key in FAILURE_CATEGORY_LABELS:
                    row[f"failure_{key}_count"] = metrics.get(f"failure_{key}_count", 0)
                    row[f"failure_{key}_rate"] = metrics.get(f"failure_{key}_rate", 0.0)
                rows.append(row)
            write_outputs(output_root, rows)
            log(f"[eval5] done method={method} {round_name}")
            try:
                import torch

                del summary
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                gc.collect()

    write_outputs(output_root, rows)
    log(f"[eval5] wrote summaries under {output_root}")


if __name__ == "__main__":
    main()
