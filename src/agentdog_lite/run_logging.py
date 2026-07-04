"""Run log directory helpers for TensorBoard artifacts."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


LOG_ROOT = Path("/root/autodl-tmp/agentdog_lite_summer_camp/logs")
VALID_RUN_KINDS = {"sft", "rl", "only_eval"}


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def _liwenbo_prefixed(name: str) -> str:
    safe = _safe_name(name)
    return safe if safe.startswith("李文博_") else f"李文博_{safe}"


def make_timestamped_log_dir(
    kind: str,
    name: str,
    timestamp: str | None = None,
    create: bool = True,
) -> Path:
    if kind not in VALID_RUN_KINDS:
        raise ValueError(f"Unsupported run log kind: {kind!r}")
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = LOG_ROOT / kind
    if create:
        base_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{_liwenbo_prefixed(name)}_{timestamp}"
    log_dir = base_dir / base_name
    if create:
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def set_tensorboard_log_dir_env(log_dir: Path) -> None:
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(log_dir)
