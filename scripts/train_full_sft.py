#!/usr/bin/env python
"""Full-parameter SFT with time-based full-model checkpoint evaluation."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agentdog_lite.loss_masking import IGNORE_INDEX, encode_text, render_prompt_and_full
from agentdog_lite.prompts import BINARY_SYSTEM_PROMPT, judgment_target
from agentdog_lite.run_logging import make_timestamped_log_dir, set_tensorboard_log_dir_env
from agentdog_lite.trajectory import extract_trajectory_from_instruction
from scripts.evaluate import (
    EvalOptions,
    apply_test_data_dir,
    evaluate_dataset,
    write_eval_tensorboard_metrics,
)


DATASET_NAMES = ("atbench", "rjudge")
EVAL_HISTORY_HEADER = [
    "checkpoint_id",
    "step",
    "created_at",
    "retained",
    "is_best",
    "is_latest",
    "overall_accuracy",
    "mean_macro_f1",
    "atbench_accuracy",
    "atbench_macro_f1",
    "rjudge_accuracy",
    "rjudge_macro_f1",
    "checkpoint_dir",
    "eval_dir",
]


def log(message: str) -> None:
    print(f"李文博_{message}", flush=True)


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE") or "1")


def env_rank() -> int:
    return int(os.environ.get("RANK") or "0")


def env_is_rank_zero() -> bool:
    return env_rank() == 0


def resolve_shared_run_timestamp(output_dir: Path) -> str:
    env_timestamp = os.environ.get("AGENTDOG_RUN_TIMESTAMP")
    if env_timestamp:
        return env_timestamp
    if env_world_size() <= 1:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    stamp_path = output_dir / ".active_run_timestamp"
    if env_is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(datetime.now().strftime("%Y%m%d_%H%M%S"), encoding="utf-8")
    deadline = time.monotonic() + 120
    while not stamp_path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timed out waiting for DDP run timestamp at {stamp_path}")
        time.sleep(0.2)
    return stamp_path.read_text(encoding="utf-8").strip()


def ensure_valid_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name, "")
        if not value.isdigit() or int(value) < 1:
            os.environ[name] = "8"


def require_supported_python() -> None:
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(
            "Full SFT requires Python >=3.10,<3.13. "
            f"Current interpreter is {sys.version.split()[0]}."
        )


def import_training_stack() -> dict[str, Any]:
    require_supported_python()
    ensure_valid_thread_env()
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing full-SFT dependencies in the active H800 env.") from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_target(output: Any) -> tuple[str, str]:
    if not isinstance(output, str) or not output.strip():
        raise RuntimeError("Alpaca output must be a non-empty JSON string.")
    try:
        obj = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Assistant output is not JSON: {output!r}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"Assistant output must be a JSON object: {output!r}")
    judgment = str(obj.get("judgment", "")).strip().lower()
    target = judgment_target(judgment)
    return judgment, target


def alpaca_row_to_messages(
    row: dict[str, Any],
    source_index: int,
    extract_trajectory: bool,
) -> dict[str, Any]:
    missing = [key for key in ("instruction", "input", "output") if key not in row]
    if missing:
        raise RuntimeError(f"Alpaca row {source_index} missing columns: {missing}")
    instruction = str(row["instruction"])
    user_content = (
        extract_trajectory_from_instruction(instruction)
        if extract_trajectory
        else instruction.strip()
    )
    if not extract_trajectory and row.get("input"):
        user_content = f"{user_content}\n{row['input']}".strip()
    if not user_content:
        raise RuntimeError(f"Alpaca row {source_index} produced empty user content.")
    label, target = normalize_target(row["output"])
    return {
        "source_index": source_index,
        "label": label,
        "messages": [
            {"role": "system", "content": BINARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ],
    }


def load_alpaca_sft_rows(
    path: Path,
    extract_trajectory: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected top-level list in {path}")
    rows = [
        alpaca_row_to_messages(row, idx, extract_trajectory)
        for idx, row in enumerate(data)
    ]
    labels = {row["label"] for row in rows}
    if labels != {"safe", "unsafe"}:
        raise RuntimeError(f"Expected safe/unsafe labels, got {sorted(labels)}")
    if max_samples is not None:
        rows = balanced_subset(rows, max_samples=max_samples, seed=seed)
    return rows


def balanced_subset(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {"safe": [], "unsafe": []}
    for row in rows:
        by_label[row["label"]].append(row)
    for group in by_label.values():
        rng.shuffle(group)
    half = max_samples // 2
    selected = by_label["safe"][:half] + by_label["unsafe"][: max_samples - half]
    rng.shuffle(selected)
    return selected


def stratified_train_dev_split(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < dev_ratio < 0.5:
        raise RuntimeError(f"dev_ratio must be in (0, 0.5), got {dev_ratio}")
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for label in ("safe", "unsafe"):
        group = [row for row in rows if row["label"] == label]
        if len(group) < 2:
            raise RuntimeError(f"Need at least two samples for label {label}")
        rng.shuffle(group)
        dev_count = max(1, round(len(group) * dev_ratio))
        dev_rows.extend(group[:dev_count])
        train_rows.extend(group[dev_count:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


@dataclass(frozen=True)
class EncodedRow:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    source_index: int
    label: str
    input_tokens: int
    original_input_tokens: int
    prompt_tokens: int
    assistant_tokens: int
    truncated: bool


def encode_assistant_only_no_truncation(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_len: int,
) -> EncodedRow:
    prompt_text, full_text = render_prompt_and_full(tokenizer, row["messages"])
    prompt_ids = encode_text(tokenizer, prompt_text)
    full_ids = encode_text(tokenizer, full_text)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Chat template full text is not prefixed by prompt text.")
    assistant_ids = full_ids[len(prompt_ids) :]
    if not assistant_ids:
        raise RuntimeError(f"Row {row['source_index']} produced zero assistant tokens.")
    original_input_tokens = len(full_ids)
    truncated = False
    if len(full_ids) > max_seq_len:
        max_prompt_len = max_seq_len - len(assistant_ids)
        marker_ids = encode_text(tokenizer, "\n[TRUNCATED_MIDDLE_FOR_CONTEXT_WINDOW]\n")
        available = max_prompt_len - len(marker_ids)
        if available <= 1:
            raise RuntimeError(
                "Assistant target is too long to preserve during prompt truncation. "
                f"row={row['source_index']} assistant_tokens={len(assistant_ids)} "
                f"max_seq_len={max_seq_len}"
            )
        head_len = max(1, int(round(available * 0.30)))
        tail_len = max(1, available - head_len)
        prompt_ids = prompt_ids[:head_len] + marker_ids + prompt_ids[-tail_len:]
        full_ids = prompt_ids + assistant_ids
        truncated = True
        if len(full_ids) > max_seq_len:
            raise RuntimeError(
                "Prompt truncation failed to fit max_seq_len. "
                f"row={row['source_index']} tokens={len(full_ids)} max_seq_len={max_seq_len}"
            )
    labels = [IGNORE_INDEX] * len(prompt_ids) + assistant_ids
    return EncodedRow(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        source_index=int(row["source_index"]),
        label=str(row["label"]),
        input_tokens=len(full_ids),
        original_input_tokens=original_input_tokens,
        prompt_tokens=len(prompt_ids),
        assistant_tokens=len(assistant_ids),
        truncated=truncated,
    )


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return int(round(sorted_values[low] * (1 - frac) + sorted_values[high] * frac))


def token_summary(encoded: list[EncodedRow]) -> dict[str, Any]:
    lengths = sorted(row.input_tokens for row in encoded)
    original_lengths = sorted(row.original_input_tokens for row in encoded)
    assistant_lengths = sorted(row.assistant_tokens for row in encoded)
    total = len(lengths)
    return {
        "num_samples": total,
        "truncated_count": sum(row.truncated for row in encoded),
        "input_tokens": {
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": lengths[-1] if lengths else 0,
        },
        "original_input_tokens": {
            "p50": percentile(original_lengths, 0.50),
            "p90": percentile(original_lengths, 0.90),
            "p95": percentile(original_lengths, 0.95),
            "p99": percentile(original_lengths, 0.99),
            "max": original_lengths[-1] if original_lengths else 0,
        },
        "assistant_tokens": {
            "p50": percentile(assistant_lengths, 0.50),
            "p90": percentile(assistant_lengths, 0.90),
            "max": assistant_lengths[-1] if assistant_lengths else 0,
        },
        "over_8192_ratio": sum(length > 8192 for length in original_lengths) / total if total else 0.0,
        "over_16384_ratio": sum(length > 16384 for length in original_lengths) / total if total else 0.0,
    }


class FullSftDataset:
    def __init__(self, encoded_rows: list[EncodedRow]) -> None:
        self.items = [
            {
                "input_ids": row.input_ids,
                "attention_mask": row.attention_mask,
                "labels": row.labels,
            }
            for row in encoded_rows
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any, torch: Any, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.torch = torch
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)
        return {key: self.torch.tensor(value, dtype=self.torch.long) for key, value in batch.items()}


def assert_effective_batch(config: dict[str, Any]) -> int:
    training = config["training"]
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    effective = (
        int(training["per_device_train_batch_size"])
        * int(training["gradient_accumulation_steps"])
        * world_size
    )
    expected = int(config["expected_effective_batch_size"])
    if effective != expected:
        raise RuntimeError(f"Effective batch size mismatch: got {effective}, expected {expected}.")
    return effective


def training_args_kwargs(training_args_cls: Any, config: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    training = dict(config["training"])
    kwargs: dict[str, Any] = {
        "output_dir": str(config["output_dir"]),
        "run_name": config["run_name"],
        "logging_dir": str(log_dir),
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "save_strategy": "no",
        "save_safetensors": True,
        **training,
    }
    kwargs["report_to"] = ["tensorboard"]
    kwargs["logging_dir"] = str(log_dir)
    kwargs["save_strategy"] = "no"
    signature = inspect.signature(training_args_cls.__init__)
    if "eval_strategy" in kwargs and "eval_strategy" not in signature.parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    accepted = set(signature.parameters)
    if "kwargs" not in accepted:
        for key in sorted(key for key in kwargs if key not in accepted):
            log(f"[train] WARNING: TrainingArguments does not accept {key}; dropping it.")
            kwargs.pop(key)
    return kwargs


def eval_options_from_config(config: dict[str, Any]) -> EvalOptions:
    eval_cfg = config["eval"]
    return EvalOptions(
        eval_batch_size=int(eval_cfg.get("eval_batch_size", 64)),
        max_input_tokens=int(eval_cfg.get("max_input_tokens", 16384)),
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 32)),
        sort_by_length=bool(eval_cfg.get("sort_by_length", True)),
        auto_reduce_batch_on_oom=bool(eval_cfg.get("auto_reduce_batch_on_oom", True)),
        torch_dtype=str(eval_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=str(eval_cfg.get("attn_implementation", "sdpa")),
    )


def combined_eval_scores(datasets: dict[str, dict[str, Any]]) -> tuple[float, float]:
    total = 0
    correct = 0.0
    macro_values = []
    for metrics in datasets.values():
        samples = int(metrics["num_samples"])
        total += samples
        correct += float(metrics["accuracy"]) * samples
        macro_values.append(float(metrics["macro_f1"]))
    return (correct / total if total else 0.0, sum(macro_values) / len(macro_values))


def retained_checkpoint_ids(entries: list[dict[str, Any]]) -> set[str]:
    candidates = [entry for entry in entries if entry.get("eval_succeeded") and entry.get("checkpoint_dir")]
    if not candidates:
        return set()
    best = max(
        candidates,
        key=lambda entry: (
            float(entry.get("overall_accuracy", 0.0)),
            float(entry.get("mean_macro_f1", 0.0)),
            int(entry.get("step", 0)),
        ),
    )
    latest = max(candidates, key=lambda entry: (str(entry.get("created_at", "")), int(entry.get("step", 0))))
    return {str(best["checkpoint_id"]), str(latest["checkpoint_id"])}


def checkpoint_path_is_safe(path: Path, checkpoint_root: Path) -> bool:
    try:
        resolved = path.resolve()
        root = checkpoint_root.resolve()
    except FileNotFoundError:
        return False
    return resolved != root and root in resolved.parents


def write_csv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(header: list[str], rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        values = []
        for key in header:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


class CheckpointEvalManager:
    def __init__(
        self,
        config: dict[str, Any],
        tokenizer: Any,
        torch: Any,
        eval_options: EvalOptions,
        sft_log_dir: Path,
        eval_limit: int | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.torch = torch
        self.eval_options = eval_options
        self.eval_limit = eval_limit
        self.output_dir = Path(config["output_dir"])
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.eval_root = self.output_dir / "evals"
        self.registry_path = self.output_dir / "checkpoint_registry.json"
        self.history_csv = self.output_dir / "eval_history.csv"
        self.history_json = self.output_dir / "eval_history.json"
        self.history_md = self.output_dir / "eval_history.md"
        self.sft_writer = None
        self.sft_log_dir = sft_log_dir
        self.eval_log_dir = make_timestamped_log_dir(
            "only_eval",
            f"{config['run_name']}_checkpoint_eval",
            timestamp=config.get("run_timestamp"),
            create=False,
        )
        self.entries: list[dict[str, Any]] = []
        self.last_checkpoint_step: int | None = None

    def distributed_initialized(self) -> bool:
        dist = getattr(self.torch, "distributed", None)
        return bool(dist and dist.is_available() and dist.is_initialized())

    def is_rank_zero(self) -> bool:
        if self.distributed_initialized():
            return self.torch.distributed.get_rank() == 0
        return env_is_rank_zero()

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        if not self.distributed_initialized():
            return value
        values = [value]
        self.torch.distributed.broadcast_object_list(values, src=src)
        return values[0]

    def distributed_barrier(self) -> None:
        if self.distributed_initialized():
            self.torch.distributed.barrier()

    def set_sft_writer(self, writer: Any) -> None:
        self.sft_writer = writer

    def save_and_evaluate(self, trainer: Any, state: Any, tag: str) -> dict[str, Any]:
        if not self.is_rank_zero():
            raise RuntimeError("Only rank0 is allowed to save checkpoints or run eval.")
        step = int(state.global_step)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = (
            f"step_{step}_{timestamp}"
            if tag == "step"
            else f"{tag}_step_{step}_{timestamp}"
        )
        checkpoint_dir = self.checkpoint_root / checkpoint_id
        eval_dir = self.eval_root / checkpoint_id
        log(f"[checkpoint] saving full model to {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        trainer.save_model(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        write_json(
            checkpoint_dir / "checkpoint_metadata.json",
            {
                "checkpoint_id": checkpoint_id,
                "step": step,
                "tag": tag,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "note": "Full model weights only; optimizer state intentionally not saved.",
            },
        )

        entry = {
            "checkpoint_id": checkpoint_id,
            "step": step,
            "tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_dir": str(checkpoint_dir),
            "eval_dir": str(eval_dir),
            "eval_succeeded": False,
            "retained": True,
            "is_best": False,
            "is_latest": False,
        }
        try:
            eval_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            summary = self.evaluate_current_model(
                model=eval_model,
                checkpoint_dir=checkpoint_dir,
                eval_dir=eval_dir,
                checkpoint_id=checkpoint_id,
                step=step,
            )
            overall_accuracy, mean_macro_f1 = combined_eval_scores(summary["datasets"])
            entry.update(
                {
                    "eval_succeeded": True,
                    "overall_accuracy": overall_accuracy,
                    "mean_macro_f1": mean_macro_f1,
                    "summary_path": str(eval_dir / "summary.json"),
                    "tensorboard_log_dir": summary["tensorboard_log_dir"],
                    "datasets": summary["datasets"],
                }
            )
            self.last_checkpoint_step = step
            log(
                "[checkpoint] eval complete "
                f"id={checkpoint_id} overall_acc={overall_accuracy:.6f} "
                f"mean_macro_f1={mean_macro_f1:.6f}"
            )
        finally:
            self.entries.append(entry)
            self.enforce_retention()
            self.write_registry_and_history()
        return entry

    def evaluate_current_model(
        self,
        model: Any,
        checkpoint_dir: Path,
        eval_dir: Path,
        checkpoint_id: str,
        step: int,
    ) -> dict[str, Any]:
        from torch.utils.tensorboard import SummaryWriter

        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_config = {
            "datasets": dict(self.config["eval"]["datasets"]),
        }
        apply_test_data_dir(eval_config, self.config["eval"].get("test_data_dir"))
        log_dir = self.eval_log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        stack = {"torch": self.torch}
        was_training = model.training
        old_padding_side = self.tokenizer.padding_side
        model.eval()
        self.tokenizer.padding_side = "left"
        dataset_metrics = {}
        actual_batch_sizes = []
        try:
            for dataset_name in DATASET_NAMES:
                dataset_path = Path(eval_config["datasets"][dataset_name])
                metrics, actual_batch_size, _timing, failure_analysis = evaluate_dataset(
                    stack=stack,
                    tokenizer=self.tokenizer,
                    model=model,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    output_dir=eval_dir,
                    method=self.config["run_name"],
                    limit=self.eval_limit,
                    options=self.eval_options,
                )
                dataset_metrics[dataset_name] = metrics
                actual_batch_sizes.append(actual_batch_size)
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{dataset_name}/{metric_name}", value, step)
                        if self.sft_writer is not None:
                            self.sft_writer.add_scalar(
                                f"checkpoint_eval/{dataset_name}/{metric_name}",
                                value,
                                step,
                            )
                write_eval_tensorboard_metrics(
                    writer=writer,
                    method=self.config["run_name"],
                    dataset_name=dataset_name,
                    metrics=metrics,
                    actual_batch_size=actual_batch_size,
                    failure_analysis=failure_analysis,
                    step=step,
                )
            overall_accuracy, mean_macro_f1 = combined_eval_scores(dataset_metrics)
            writer.add_scalar("combined/overall_accuracy", overall_accuracy, step)
            writer.add_scalar("combined/mean_macro_f1", mean_macro_f1, step)
            if self.sft_writer is not None:
                self.sft_writer.add_scalar("checkpoint_eval/combined/overall_accuracy", overall_accuracy, step)
                self.sft_writer.add_scalar("checkpoint_eval/combined/mean_macro_f1", mean_macro_f1, step)
                self.sft_writer.flush()
            writer.add_text(
                "eval/config",
                json.dumps(
                    {
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_dir": str(checkpoint_dir),
                        "eval_options": self.eval_options.__dict__,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                step,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()
            writer.flush()
            writer.close()

        summary = {
            "method": self.config["run_name"],
            "model_path": str(checkpoint_dir),
            "adapter_path": None,
            "output_dir": str(eval_dir),
            "eval_batch_size_requested": self.eval_options.eval_batch_size,
            "eval_batch_size_actual": min(actual_batch_sizes) if actual_batch_sizes else self.eval_options.eval_batch_size,
            "max_input_tokens": self.eval_options.max_input_tokens,
            "max_new_tokens": self.eval_options.max_new_tokens,
            "torch_dtype": self.eval_options.torch_dtype,
            "attn_implementation": self.eval_options.attn_implementation,
            "tensorboard_log_dir": str(log_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "datasets": dataset_metrics,
        }
        write_json(eval_dir / "summary.json", summary)
        return summary

    def enforce_retention(self) -> None:
        keep_ids = retained_checkpoint_ids(self.entries)
        latest_id = None
        best_id = None
        candidates = [entry for entry in self.entries if entry.get("eval_succeeded")]
        if candidates:
            latest_id = max(candidates, key=lambda entry: (str(entry.get("created_at", "")), int(entry.get("step", 0))))["checkpoint_id"]
            best_id = max(
                candidates,
                key=lambda entry: (
                    float(entry.get("overall_accuracy", 0.0)),
                    float(entry.get("mean_macro_f1", 0.0)),
                    int(entry.get("step", 0)),
                ),
            )["checkpoint_id"]
        for entry in self.entries:
            checkpoint_id = str(entry.get("checkpoint_id"))
            entry["is_best"] = checkpoint_id == best_id
            entry["is_latest"] = checkpoint_id == latest_id
            if checkpoint_id in keep_ids or not entry.get("checkpoint_dir"):
                entry["retained"] = True
                continue
            path = Path(entry["checkpoint_dir"])
            if path.exists() and checkpoint_path_is_safe(path, self.checkpoint_root):
                log(f"[checkpoint] deleting non-best/non-latest checkpoint {path}")
                shutil.rmtree(path)
            entry["retained"] = False
            entry["deleted_at"] = datetime.now(timezone.utc).isoformat()
        self.update_checkpoint_links(best_id=best_id, latest_id=latest_id)

    def update_checkpoint_links(self, best_id: str | None, latest_id: str | None) -> None:
        id_to_path = {
            str(entry["checkpoint_id"]): Path(entry["checkpoint_dir"])
            for entry in self.entries
            if entry.get("checkpoint_dir") and Path(entry["checkpoint_dir"]).exists()
        }
        for link_name, checkpoint_id in (("best_checkpoint", best_id), ("latest_checkpoint", latest_id)):
            link = self.output_dir / link_name
            if link.is_symlink() or link.is_file():
                link.unlink()
            if checkpoint_id and checkpoint_id in id_to_path:
                link.symlink_to(id_to_path[checkpoint_id].resolve(), target_is_directory=True)

    def history_rows(self) -> list[dict[str, Any]]:
        rows = []
        for entry in self.entries:
            datasets = entry.get("datasets", {})
            row = {
                "checkpoint_id": entry.get("checkpoint_id"),
                "step": entry.get("step"),
                "created_at": entry.get("created_at"),
                "retained": entry.get("retained"),
                "is_best": entry.get("is_best"),
                "is_latest": entry.get("is_latest"),
                "overall_accuracy": entry.get("overall_accuracy", ""),
                "mean_macro_f1": entry.get("mean_macro_f1", ""),
                "checkpoint_dir": entry.get("checkpoint_dir"),
                "eval_dir": entry.get("eval_dir"),
            }
            for dataset in DATASET_NAMES:
                metrics = datasets.get(dataset, {})
                row[f"{dataset}_accuracy"] = metrics.get("accuracy", "")
                row[f"{dataset}_macro_f1"] = metrics.get("macro_f1", "")
            rows.append(row)
        return rows

    def write_registry_and_history(self) -> None:
        write_json(self.registry_path, {"checkpoints": self.entries})
        rows = self.history_rows()
        write_csv(self.history_csv, rows, EVAL_HISTORY_HEADER)
        write_json(self.history_json, rows)
        self.history_md.write_text(markdown_table(EVAL_HISTORY_HEADER, rows), encoding="utf-8")


class TimeCheckpointCallback:
    def __init__(self, trainer_callback_cls: Any, manager: CheckpointEvalManager, interval_seconds: float) -> None:
        self.manager = manager
        self.interval_seconds = interval_seconds
        self.last_checkpoint_time = time.monotonic()
        self.in_progress = False

        class _Callback(trainer_callback_cls):
            def on_step_end(inner_self, args, state, control, **kwargs):  # noqa: ANN001
                if state.global_step <= 0:
                    return control
                should_checkpoint = False
                if self.manager.is_rank_zero():
                    should_checkpoint = (
                        not self.in_progress
                        and time.monotonic() - self.last_checkpoint_time >= self.interval_seconds
                    )
                should_checkpoint = bool(self.manager.broadcast_object(should_checkpoint))
                if not should_checkpoint:
                    return control
                self.in_progress = True
                self.manager.distributed_barrier()
                try:
                    if self.manager.is_rank_zero():
                        self.manager.save_and_evaluate(self.trainer, state, tag="step")
                        self.last_checkpoint_time = time.monotonic()
                    self.manager.distributed_barrier()
                finally:
                    self.in_progress = False
                return control

        self.callback = _Callback()
        self.trainer = None


def save_resolved_config(config: dict[str, Any], output_dir: Path, config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["config_path"] = str(config_path)
    write_json(output_dir / "training_config_resolved.json", resolved)
    (output_dir / "training_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(config_path, output_dir / "training_config_input.yaml")


def print_training_header(
    torch: Any,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    effective_batch: int,
) -> None:
    training = config["training"]
    log("[train] full-parameter SFT configuration")
    log(f"[train] model_path={config['base_model']}")
    log(f"[train] train_samples={len(train_rows)} dev_samples={len(dev_rows)}")
    log(f"[train] max_seq_len={config['max_seq_len']}")
    log(f"[train] per_device_train_batch_size={training['per_device_train_batch_size']}")
    log(f"[train] gradient_accumulation_steps={training['gradient_accumulation_steps']}")
    log(f"[train] effective_batch={effective_batch}")
    log(f"[train] checkpoint_interval_seconds={config['checkpoint_interval_seconds']}")
    log(f"[train] output_dir={config['output_dir']}")
    log(f"[train] tensorboard_log_dir={config['tensorboard_log_dir']}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"[train] gpu={props.name} total_memory={props.total_memory / 1024**3:.2f}GiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--checkpoint-interval-seconds", type=float)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--dry-run-preprocess", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    if args.checkpoint_interval_seconds is not None:
        config["checkpoint_interval_seconds"] = args.checkpoint_interval_seconds
    output_dir = Path(config["output_dir"])
    config["output_dir"] = str(output_dir)
    effective_batch = assert_effective_batch(config)
    run_timestamp = resolve_shared_run_timestamp(output_dir)
    config["run_timestamp"] = run_timestamp

    stack = import_training_stack()
    torch = stack["torch"]
    if config["training"].get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    log_dir = make_timestamped_log_dir(
        "sft",
        config["run_name"],
        timestamp=run_timestamp,
        create=env_is_rank_zero(),
    )
    set_tensorboard_log_dir_env(log_dir)
    config["tensorboard_log_dir"] = str(log_dir)
    config["log_kind"] = "sft"
    if env_is_rank_zero():
        save_resolved_config(config, output_dir, config_path)

    tokenizer = stack["AutoTokenizer"].from_pretrained(config["base_model"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_alpaca_sft_rows(
        Path(config["train_file"]),
        extract_trajectory=bool(config.get("extract_trajectory", True)),
        max_samples=args.max_samples,
        seed=int(config.get("seed", 42)),
    )
    train_rows, dev_rows = stratified_train_dev_split(
        rows,
        dev_ratio=float(config.get("dev_ratio", 0.10)),
        seed=int(config.get("seed", 42)),
    )
    train_encoded = [
        encode_assistant_only_no_truncation(tokenizer, row, int(config["max_seq_len"]))
        for row in train_rows
    ]
    dev_encoded = [
        encode_assistant_only_no_truncation(tokenizer, row, int(config["max_seq_len"]))
        for row in dev_rows
    ]
    token_length_summary = {
        "max_seq_len": int(config["max_seq_len"]),
        "all_samples": len(rows),
        "train": token_summary(train_encoded),
        "dev": token_summary(dev_encoded),
    }
    if env_is_rank_zero():
        write_json(output_dir / "token_length_summary.json", token_length_summary)
    split_summary = {
        "train_samples": len(train_rows),
        "dev_samples": len(dev_rows),
        "train_label_counts": {
            label: sum(row["label"] == label for row in train_rows)
            for label in ("safe", "unsafe")
        },
        "dev_label_counts": {
            label: sum(row["label"] == label for row in dev_rows)
            for label in ("safe", "unsafe")
        },
        "dry_run_preprocess": bool(args.dry_run_preprocess),
    }
    if env_is_rank_zero():
        write_json(output_dir / "data_split_summary.json", split_summary)
    if args.dry_run_preprocess:
        if env_is_rank_zero():
            log(f"[dry-run] wrote {output_dir / 'token_length_summary.json'}")
            log(f"[dry-run] wrote {output_dir / 'data_split_summary.json'}")
        return

    dtype = torch.bfloat16 if config["training"].get("bf16", False) else torch.float16
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": str(config.get("attn_implementation", "sdpa")),
    }
    model = stack["AutoModelForCausalLM"].from_pretrained(config["base_model"], **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if env_is_rank_zero():
        print_training_header(torch, config, train_rows, dev_rows, effective_batch)
    train_args = stack["TrainingArguments"](
        **training_args_kwargs(stack["TrainingArguments"], config, log_dir)
    )
    manager = CheckpointEvalManager(
        config=config,
        tokenizer=tokenizer,
        torch=torch,
        eval_options=eval_options_from_config(config),
        sft_log_dir=log_dir,
        eval_limit=args.eval_limit,
    )
    callback_wrapper = TimeCheckpointCallback(
        stack["TrainerCallback"],
        manager=manager,
        interval_seconds=float(config.get("checkpoint_interval_seconds", 1800)),
    )
    trainer = stack["Trainer"](
        model=model,
        args=train_args,
        train_dataset=FullSftDataset(train_encoded),
        eval_dataset=FullSftDataset(dev_encoded),
        data_collator=AssistantOnlyCollator(tokenizer, torch),
        callbacks=[callback_wrapper.callback],
    )
    callback_wrapper.trainer = trainer
    manager.set_sft_writer(None)
    sft_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        if manager.is_rank_zero():
            sft_writer = SummaryWriter(log_dir=str(log_dir))
            manager.set_sft_writer(sft_writer)
            sft_writer.add_text(
                "train/config",
                json.dumps(config, ensure_ascii=False, indent=2),
                0,
            )
            sft_writer.add_text(
                "train/token_length_summary",
                json.dumps(token_length_summary, ensure_ascii=False, indent=2),
                0,
            )
        trainer.train()
        if manager.is_rank_zero() and manager.last_checkpoint_step != int(trainer.state.global_step):
            manager.save_and_evaluate(trainer, trainer.state, tag="final")
        manager.distributed_barrier()
    finally:
        if sft_writer is not None:
            sft_writer.flush()
            sft_writer.close()
    if manager.is_rank_zero():
        log(f"[train] finished full SFT. Registry: {manager.registry_path}")


if __name__ == "__main__":
    main()
