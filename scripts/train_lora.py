#!/usr/bin/env python
"""Train a PEFT LoRA adapter for AgentDoG-Lite with assistant-only loss."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdog_lite.loss_masking import IGNORE_INDEX, encode_assistant_only
from agentdog_lite.run_logging import make_timestamped_log_dir, set_tensorboard_log_dir_env


def log(message: str) -> None:
    print(f"李文博_{message}")


def require_supported_python() -> None:
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(
            "Training requires Python >=3.10,<3.13. "
            f"Current interpreter is {sys.version.split()[0]}. "
            "Reuse the current H800 environment when possible."
        )


def import_training_stack() -> dict[str, Any]:
    require_supported_python()
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:  # noqa: BLE001 - make dependency failure explicit.
        raise RuntimeError(
            "Missing or incompatible training dependencies. Install in the current env with:\n"
            "  python -m pip install -U 'transformers>=4.57.0' 'accelerate>=1.10.0' "
            "'datasets>=3.0.0' 'peft>=0.17.0' 'trl>=0.24.0'\n"
            "  python -m pip install -e ."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "Trainer": Trainer,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no} missing messages")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_resolved_config(config: dict[str, Any], output_dir: Path, config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["config_path"] = str(config_path)
    commit_path = Path("third_party/AgentDoG_COMMIT.txt")
    if commit_path.exists():
        resolved["agentdog_reference_commit"] = commit_path.read_text(encoding="utf-8").strip()
    (output_dir / "training_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(config_path, output_dir / "training_config_input.yaml")


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
        raise RuntimeError(
            f"Effective batch size mismatch: got {effective}, expected {expected}. "
            "Adjust per_device_train_batch_size or gradient_accumulation_steps explicitly."
        )
    return effective


class AssistantOnlyDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
        self.items = []
        self.truncated_count = 0
        for row in rows:
            encoded = encode_assistant_only(tokenizer, row["messages"], max_seq_len=max_seq_len)
            self.items.append(
                {
                    "input_ids": encoded.input_ids,
                    "attention_mask": encoded.attention_mask,
                    "labels": encoded.labels,
                    "length": len(encoded.input_ids),
                }
            )
            self.truncated_count += int(encoded.truncated)

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
        return {
            key: self.torch.tensor(value, dtype=self.torch.long)
            for key, value in batch.items()
        }


def training_args_kwargs(
    training_args_cls: Any,
    config: dict[str, Any],
    logging_dir: Path,
) -> dict[str, Any]:
    training = dict(config["training"])
    output_dir = config["output_dir"]
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": config.get("run_name", Path(output_dir).name),
        "logging_dir": str(logging_dir),
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        **training,
    }
    signature = inspect.signature(training_args_cls.__init__)
    if "eval_strategy" in kwargs and "eval_strategy" not in signature.parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    kwargs["logging_dir"] = str(logging_dir)
    kwargs["report_to"] = ["tensorboard"]

    accepted = set(signature.parameters)
    if "kwargs" in accepted:
        return kwargs
    dropped = sorted(key for key in kwargs if key not in accepted)
    for key in dropped:
        print(f"[train] WARNING: TrainingArguments does not accept {key}; dropping it.")
        kwargs.pop(key)
    return kwargs


def load_model_with_sdpa(stack: dict[str, Any], config: dict[str, Any], dtype: Any) -> Any:
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    try:
        return stack["AutoModelForCausalLM"].from_pretrained(config["base_model"], **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return stack["AutoModelForCausalLM"].from_pretrained(config["base_model"], **kwargs)


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def print_training_header(
    torch: Any,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    effective_batch: int,
    trainable_params: int,
    total_params: int,
) -> None:
    training = config["training"]
    lora = config["lora"]
    log("[train] H800 SFT training configuration")
    log(f"[train] model_path={config['base_model']}")
    log(f"[train] train_samples={len(train_rows)} dev_samples={len(dev_rows)}")
    log(f"[train] max_seq_len={config['max_seq_len']}")
    log(f"[train] per_device_train_batch_size={training['per_device_train_batch_size']}")
    log(f"[train] gradient_accumulation_steps={training['gradient_accumulation_steps']}")
    log(f"[train] effective_batch={effective_batch}")
    log(f"[train] lora_rank={lora['r']} lora_alpha={lora['lora_alpha']}")
    log(f"[train] trainable_params={trainable_params} total_params={total_params}")
    log(f"[train] tensorboard_log_dir={config['tensorboard_log_dir']}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            "李文博_[train] gpu="
            f"{props.name} total_memory={props.total_memory / 1024**3:.2f}GiB"
        )
    else:
        log("[train] gpu=unavailable")


def print_oom_guidance() -> None:
    print(
        "李文博_[train] CUDA OOM. Keep expected_effective_batch_size=128. "
        "If batch=4/accum=32 also OOMs, use:\n"
        "  per_device_train_batch_size: 2\n"
        "  gradient_accumulation_steps: 64\n"
        "First fallback remains:\n"
        "  per_device_train_batch_size: 4\n"
        "  gradient_accumulation_steps: 32\n"
        "Do not silently reduce the effective batch.",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    effective_batch = assert_effective_batch(config)
    stack = import_training_stack()
    torch = stack["torch"]

    if config["training"].get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(config["output_dir"])
    log_dir = make_timestamped_log_dir("sft", config.get("run_name", output_dir.name))
    set_tensorboard_log_dir_env(log_dir)
    config["tensorboard_log_dir"] = str(log_dir)
    config["log_kind"] = "sft"
    save_resolved_config(config, output_dir, config_path)

    tokenizer = stack["AutoTokenizer"].from_pretrained(config["base_model"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(config["train_file"]))
    dev_rows = read_jsonl(Path(config["dev_file"]))
    train_dataset = AssistantOnlyDataset(train_rows, tokenizer, int(config["max_seq_len"]))
    dev_dataset = AssistantOnlyDataset(dev_rows, tokenizer, int(config["max_seq_len"]))

    dtype = torch.bfloat16 if config["training"].get("bf16", False) else torch.float16
    model = load_model_with_sdpa(stack, config, dtype)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = config["lora"]
    peft_config = stack["LoraConfig"](
        task_type="CAUSAL_LM",
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = stack["get_peft_model"](model, peft_config)
    trainable_params, total_params = count_parameters(model)

    print_training_header(
        torch=torch,
        config=config,
        train_rows=train_rows,
        dev_rows=dev_rows,
        effective_batch=effective_batch,
        trainable_params=trainable_params,
        total_params=total_params,
    )
    print(
        f"李文博_[train] tokenized_truncated_count train={train_dataset.truncated_count} "
        f"dev={dev_dataset.truncated_count}"
    )

    train_args = stack["TrainingArguments"](
        **training_args_kwargs(stack["TrainingArguments"], config, log_dir)
    )
    trainer = stack["Trainer"](
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=AssistantOnlyCollator(tokenizer, torch),
    )
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print_oom_guidance()
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print_oom_guidance()
        raise
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"[train] adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
