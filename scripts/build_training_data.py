#!/usr/bin/env python
"""Build AgentDoG-Lite chat-format SFT data.

The summer camp test set is intentionally not read here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdog_lite.prompts import (
    BINARY_SYSTEM_PROMPT,
    DIAGNOSTIC_SYSTEM_PROMPT,
    judgment_target,
)
from agentdog_lite.trajectory import extract_trajectory_from_instruction, normalize_gold_label


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_finegrained_target(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = key.strip().lower().replace(" ", "_")
        values[key_norm] = value.strip()
    required = ["risk_source", "failure_mode", "real_world_harm"]
    missing = [key for key in required if key not in values or not values[key]]
    if missing:
        raise ValueError(f"Fine-grained output missing {missing}: {text!r}")
    return {
        "judgment": "unsafe",
        "risk_source": values["risk_source"],
        "failure_mode": values["failure_mode"],
        "real_world_harm": values["real_world_harm"],
    }


def make_messages(system: str, user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def binary_record(uid: str, trajectory: str, judgment: str, source_config: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "messages": make_messages(
            BINARY_SYSTEM_PROMPT,
            trajectory,
            judgment_target(judgment),
        ),
        "task_type": "binary_judgment",
        "judgment": judgment,
        "source_config": source_config,
    }


def diagnostic_record(
    uid: str,
    trajectory: str,
    target: dict[str, str],
    source_config: str,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "messages": make_messages(
            DIAGNOSTIC_SYSTEM_PROMPT,
            trajectory,
            json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        ),
        "task_type": "diagnostic_auxiliary",
        "judgment": target["judgment"],
        "source_config": source_config,
    }


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("task_type", "")),
        str(row.get("source_config", "")),
        str(row.get("judgment", "")),
    )


def stratified_sample(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    key_fn=group_key,
) -> list[dict[str, Any]]:
    if n > len(rows):
        raise ValueError(f"Cannot sample {n} rows from only {len(rows)} rows")
    if n == len(rows):
        sampled = rows[:]
        rng.shuffle(sampled)
        return sampled

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)

    raw_alloc = {key: n * len(group) / len(rows) for key, group in groups.items()}
    alloc = {key: int(value) for key, value in raw_alloc.items()}
    remaining = n - sum(alloc.values())
    by_remainder = sorted(raw_alloc, key=lambda key: raw_alloc[key] - alloc[key], reverse=True)
    for key in by_remainder[:remaining]:
        alloc[key] += 1

    sampled: list[dict[str, Any]] = []
    for key, group in groups.items():
        group_copy = group[:]
        rng.shuffle(group_copy)
        sampled.extend(group_copy[: alloc[key]])
    rng.shuffle(sampled)
    return sampled


def stratified_split(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for group in groups.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        dev_n = round(len(shuffled) * dev_ratio)
        if len(shuffled) > 1 and dev_ratio > 0:
            dev_n = max(1, min(len(shuffled) - 1, dev_n))
        dev_rows.extend(shuffled[:dev_n])
        train_rows.extend(shuffled[dev_n:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


def load_binary(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)
    records: list[dict[str, Any]] = []
    for idx, example in enumerate(rows):
        judgment = normalize_gold_label(example["output"])
        trajectory = extract_trajectory_from_instruction(example["instruction"])
        records.append(
            binary_record(
                uid=f"binary_{idx}",
                trajectory=trajectory,
                judgment=judgment,
                source_config="AgentDoG-BinarySafety",
            )
        )
    return records


def load_diagnostic(
    finegrained_path: Path,
    binary_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_json(finegrained_path)
    records: list[dict[str, Any]] = []
    for idx, example in enumerate(rows):
        trajectory = extract_trajectory_from_instruction(example["instruction"])
        target = parse_finegrained_target(example["output"])
        records.append(
            diagnostic_record(
                uid=f"fg_{idx}",
                trajectory=trajectory,
                target=target,
                source_config="AgentDoG-FineGrainedTaxonomy",
            )
        )

    for row in binary_records:
        if row["judgment"] != "safe":
            continue
        trajectory = row["messages"][1]["content"]
        records.append(
            diagnostic_record(
                uid=f"{row['uid']}_diagnostic_safe",
                trajectory=trajectory,
                target={
                    "judgment": "safe",
                    "risk_source": "none",
                    "failure_mode": "none",
                    "real_world_harm": "none",
                },
                source_config="AgentDoG-BinarySafety",
            )
        )
    return records


def load_hard_boundary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_json(path)
    records = []
    for example in rows:
        judgment = normalize_gold_label(example["judgment"])
        records.append(
            {
                **binary_record(
                    uid=example["uid"],
                    trajectory=example["trajectory"],
                    judgment=judgment,
                    source_config="hard_boundary",
                ),
                "risk_source": example.get("risk_source"),
                "failure_mode": example.get("failure_mode"),
                "real_world_harm": example.get("real_world_harm"),
            }
        )
    return records


def build_mixed(
    binary_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    hard_rows: list[dict[str, Any]],
    rng: random.Random,
    min_hard_for_ratio: int,
    min_total_with_hard: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_total_without_hard = min(
        (len(binary_rows) * 100) // 75,
        (len(diagnostic_rows) * 100) // 25,
    )
    can_use_hard = len(hard_rows) >= min_hard_for_ratio
    max_total_with_hard = 0
    if can_use_hard:
        max_total_with_hard = min(
            (len(binary_rows) * 100) // 70,
            (len(diagnostic_rows) * 100) // 20,
            (len(hard_rows) * 100) // 10,
        )
    use_hard = can_use_hard and max_total_with_hard >= min_total_with_hard

    if use_hard:
        total = max_total_with_hard
        hard_count = total * 10 // 100
        diagnostic_count = total * 20 // 100
        binary_count = total - hard_count - diagnostic_count
        counts = {
            "binary_judgment": binary_count,
            "diagnostic_auxiliary": diagnostic_count,
            "hard_boundary": hard_count,
        }
        rows = (
            stratified_sample(binary_rows, counts["binary_judgment"], rng)
            + stratified_sample(diagnostic_rows, counts["diagnostic_auxiliary"], rng)
            + stratified_sample(hard_rows, counts["hard_boundary"], rng)
        )
        ratio_mode = "70_binary_20_diagnostic_10_hard"
    else:
        total = max_total_without_hard
        diagnostic_count = total * 25 // 100
        binary_count = total - diagnostic_count
        counts = {
            "binary_judgment": binary_count,
            "diagnostic_auxiliary": diagnostic_count,
            "hard_boundary": 0,
        }
        rows = (
            stratified_sample(binary_rows, counts["binary_judgment"], rng)
            + stratified_sample(diagnostic_rows, counts["diagnostic_auxiliary"], rng)
        )
        ratio_mode = "75_binary_25_diagnostic_hard_insufficient"

    rng.shuffle(rows)
    summary = {
        "ratio_mode": ratio_mode,
        "counts": counts,
        "available": {
            "binary_judgment": len(binary_rows),
            "diagnostic_auxiliary": len(diagnostic_rows),
            "hard_boundary": len(hard_rows),
        },
        "min_hard_for_ratio": min_hard_for_ratio,
        "min_total_with_hard": min_total_with_hard,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary-path",
        default="data/raw/agentdog_training/AgentDoG-BinarySafety/train.json",
    )
    parser.add_argument(
        "--finegrained-path",
        default="data/raw/agentdog_training/AgentDoG-FineGrainedTaxonomy/train.json",
    )
    parser.add_argument("--hard-path", default="data/hard_boundary/hard_boundary_seed.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--min-hard-for-ratio", type=int, default=100)
    parser.add_argument("--min-total-with-hard", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    binary_rows = load_binary(Path(args.binary_path))
    diagnostic_rows = load_diagnostic(Path(args.finegrained_path), binary_rows)
    hard_rows = load_hard_boundary(Path(args.hard_path))

    mixed_rows, mixed_summary = build_mixed(
        binary_rows,
        diagnostic_rows,
        hard_rows,
        rng,
        min_hard_for_ratio=args.min_hard_for_ratio,
        min_total_with_hard=args.min_total_with_hard,
    )
    train_rows, dev_rows = stratified_split(mixed_rows, args.dev_ratio, rng)

    counts = {
        "train_binary.jsonl": write_jsonl(output_dir / "train_binary.jsonl", binary_rows),
        "train_diagnostic_aux.jsonl": write_jsonl(
            output_dir / "train_diagnostic_aux.jsonl", diagnostic_rows
        ),
        "train_hard_boundary.jsonl": write_jsonl(output_dir / "train_hard_boundary.jsonl", hard_rows),
        "train_mixed.jsonl": write_jsonl(output_dir / "train_mixed.jsonl", mixed_rows),
        "train_mixed_train.jsonl": write_jsonl(output_dir / "train_mixed_train.jsonl", train_rows),
        "train_mixed_dev.jsonl": write_jsonl(output_dir / "train_mixed_dev.jsonl", dev_rows),
    }

    build_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "files": counts,
        "mixed": mixed_summary,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(build_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
