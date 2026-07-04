#!/usr/bin/env python
"""Fail-fast environment and artifact checks for AgentDoG-Lite."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_FOR_EVAL = ["torch", "transformers", "peft", "yaml", "tensorboard"]
REQUIRED_FOR_TRAIN = REQUIRED_FOR_EVAL + ["datasets", "trl", "sklearn", "pandas"]

MODEL_DIRS = [
    Path("models/Qwen3.5-0.8B"),
    Path("models/AgentDoG1.5-Qwen3.5-0.8B"),
    Path("models/AgentDoG1.5-FG-Qwen3.5-0.8B"),
]

DATA_FILES = [
    Path("data/AgentDoG1.0-Training-Data/AgentDoG-BinarySafety/train.json"),
    Path("data/AgentDoG1.0-Training-Data/AgentDoG-FineGrainedTaxonomy/train.json"),
    Path("data/2026_summer_camp_teseset/summer_camp_ATBench300.json"),
    Path("data/2026_summer_camp_teseset/summer_camp_rjudge.json"),
    Path("data/processed/train_mixed_train.jsonl"),
    Path("data/processed/train_mixed_dev.jsonl"),
]


def check_python() -> list[str]:
    issues = []
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        issues.append(f"Python must be >=3.10,<3.13, got {sys.version.split()[0]}")
    return issues


def check_modules(modules: list[str]) -> list[str]:
    issues = []
    for module in modules:
        try:
            imported = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Missing module {module}: {exc}")
            continue
        version = getattr(imported, "__version__", "unknown")
        print(f"[module] {module} {version}")
    return issues


def check_files(paths: list[Path]) -> list[str]:
    issues = []
    for path in paths:
        if not path.exists():
            issues.append(f"Missing path: {path}")
        elif path.is_file() and path.stat().st_size == 0:
            issues.append(f"Empty file: {path}")
        else:
            print(f"[file] ok {path}")
    return issues


def check_gpu() -> list[str]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return [f"Cannot import torch for GPU check: {exc}"]
    if not torch.cuda.is_available():
        return ["torch.cuda.is_available() is false"]
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"[gpu] cuda:{idx} {props.name} {props.total_memory / 1024**3:.1f} GiB")
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "train"], default="train")
    args = parser.parse_args()

    modules = REQUIRED_FOR_TRAIN if args.mode == "train" else REQUIRED_FOR_EVAL
    issues = []
    issues.extend(check_python())
    issues.extend(check_modules(modules))
    issues.extend(check_gpu())
    issues.extend(check_files(MODEL_DIRS + DATA_FILES))
    if issues:
        for issue in issues:
            print(f"[FAIL] {issue}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[ok] environment is ready for {args.mode}")


if __name__ == "__main__":
    main()
