#!/usr/bin/env python
"""Download required Hugging Face model snapshots to local model directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_SPECS = {
    "qwen35_08b_baseline": {
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "local_dir": "models/Qwen3.5-0.8B",
    },
    "agentdog15_08b_reference": {
        "repo_id": "AI45Research/AgentDoG1.5-Qwen3.5-0.8B",
        "local_dir": "models/AgentDoG1.5-Qwen3.5-0.8B",
    },
    "agentdog15_fg_teacher": {
        "repo_id": "AI45Research/AgentDoG1.5-FG-Qwen3.5-0.8B",
        "local_dir": "models/AgentDoG1.5-FG-Qwen3.5-0.8B",
    },
}


def validate_model_dir(path: Path) -> None:
    required_any = ["model.safetensors", "model.safetensors.index.json", "model-00000-of-00001.safetensors"]
    missing_core = not any((path / name).exists() for name in required_any)
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing_core:
        missing.append("model weights")
    if missing:
        raise RuntimeError(f"Incomplete model snapshot at {path}: missing {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=sorted(MODEL_SPECS),
        action="append",
        help="Download only selected model(s). Can be repeated.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    selected = args.only or list(MODEL_SPECS)
    for name in selected:
        spec = MODEL_SPECS[name]
        local_dir = Path(spec["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {name}: {spec['repo_id']} -> {local_dir}")
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="model",
            revision=args.revision,
            local_dir=str(local_dir),
            force_download=args.force_download,
        )
        validate_model_dir(local_dir)
    print("[download] all requested model snapshots are present and validated")


if __name__ == "__main__":
    main()
