#!/usr/bin/env python
"""Copy the selected final LoRA adapter to outputs/final_continued_lora_adapter."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IGNORE_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "training_args.bin",
}


def find_adapter_dir(run_dir: Path) -> Path:
    if (run_dir / "adapter_config.json").exists():
        return run_dir
    trainer_state = run_dir / "trainer_state.json"
    if trainer_state.exists():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if best and (Path(best) / "adapter_config.json").exists():
            return Path(best)
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    for checkpoint in reversed(checkpoints):
        if (checkpoint / "adapter_config.json").exists():
            return checkpoint
    raise FileNotFoundError(f"No PEFT adapter found under {run_dir}")


def ignore_large(_: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name in IGNORE_NAMES}
    ignored.update(name for name in names if name.startswith("global_step"))
    return ignored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/our_agentdog15_continued_lora")
    parser.add_argument("--output-dir", default="outputs/final_continued_lora_adapter")
    args = parser.parse_args()

    source = find_adapter_dir(Path(args.run_dir))
    target = Path(args.output_dir)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=ignore_large)
    print(f"[export] copied adapter {source} -> {target}")


if __name__ == "__main__":
    main()

