#!/usr/bin/env python
"""Train a PEFT LoRA adapter for AgentDoG-Lite."""

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


def require_supported_python() -> None:
    if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
        raise RuntimeError(
            "Training requires Python >=3.10,<3.12. "
            f"Current interpreter is {sys.version.split()[0]}. "
            "Create the conda env from environment.yml instead of using this interpreter."
        )


def import_training_stack() -> dict[str, Any]:
    require_supported_python()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except Exception as exc:  # noqa: BLE001 - make dependency failure explicit.
        raise RuntimeError(
            "Missing or incompatible training dependencies. Install with:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate agentdog-lite"
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "SFTTrainer": SFTTrainer,
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


def assert_effective_batch(config: dict[str, Any]) -> None:
    training = config["training"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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


def build_text_dataset(rows: list[dict[str, Any]], tokenizer: Any, dataset_cls: Any) -> Any:
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered.append({"text": text, "uid": row.get("uid"), "task_type": row.get("task_type")})
    return dataset_cls.from_list(rendered)


def training_args_kwargs(training_args_cls: Any, config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config["training"])
    output_dir = config["output_dir"]
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": config.get("run_name", Path(output_dir).name),
        "remove_unused_columns": False,
        **training,
    }
    signature = inspect.signature(training_args_cls.__init__)
    if "eval_strategy" in kwargs and "eval_strategy" not in signature.parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if kwargs.get("report_to") == "none":
        kwargs["report_to"] = []
    return kwargs


def sft_trainer_kwargs(stack: dict[str, Any], base_kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(stack["SFTTrainer"].__init__)
    kwargs = {k: v for k, v in base_kwargs.items() if k in signature.parameters}
    if "tokenizer" in signature.parameters and "tokenizer" not in kwargs:
        kwargs["tokenizer"] = base_kwargs["processing_class"]
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = base_kwargs["processing_class"]
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = base_kwargs["max_seq_length"]
    elif "max_length" in signature.parameters:
        kwargs["max_length"] = base_kwargs["max_seq_length"]
    if "dataset_text_field" in signature.parameters:
        kwargs["dataset_text_field"] = "text"
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    assert_effective_batch(config)
    stack = import_training_stack()
    torch = stack["torch"]

    output_dir = Path(config["output_dir"])
    save_resolved_config(config, output_dir, config_path)

    tokenizer = stack["AutoTokenizer"].from_pretrained(config["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(config["train_file"]))
    dev_rows = read_jsonl(Path(config["dev_file"]))
    train_dataset = build_text_dataset(train_rows, tokenizer, stack["Dataset"])
    dev_dataset = build_text_dataset(dev_rows, tokenizer, stack["Dataset"])

    dtype = torch.bfloat16 if config["training"].get("bf16", False) else torch.float16
    model = stack["AutoModelForCausalLM"].from_pretrained(
        config["base_model"],
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    lora_cfg = config["lora"]
    peft_config = stack["LoraConfig"](
        task_type="CAUSAL_LM",
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
    )

    train_args = stack["TrainingArguments"](**training_args_kwargs(stack["TrainingArguments"], config))
    base_trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "max_seq_length": int(config["max_seq_len"]),
        "packing": False,
    }
    trainer = stack["SFTTrainer"](**sft_trainer_kwargs(stack, base_trainer_kwargs))
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train] adapter saved to {output_dir}")


if __name__ == "__main__":
    main()

