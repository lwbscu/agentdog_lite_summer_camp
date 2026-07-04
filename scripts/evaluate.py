#!/usr/bin/env python
"""Evaluate baseline, reference, and continued-LoRA models on summer camp test sets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdog_lite.metrics import compute_binary_metrics
from agentdog_lite.parser import parse_model_output
from agentdog_lite.prompts import BINARY_SYSTEM_PROMPT
from agentdog_lite.trajectory import format_eval_trajectory, normalize_gold_label


GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_new_tokens": 32,
    "do_sample": False,
}


def require_supported_python() -> None:
    if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
        raise RuntimeError(
            "Evaluation requires Python >=3.10,<3.12 for the installed model stack. "
            f"Current interpreter is {sys.version.split()[0]}."
        )


def import_eval_stack() -> dict[str, Any]:
    require_supported_python()
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing evaluation dependencies. Install with:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate agentdog-lite"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_model(stack: dict[str, Any], model_path: str, adapter_path: str | None) -> tuple[Any, Any]:
    torch = stack["torch"]
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")
    tokenizer = stack["AutoTokenizer"].from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = stack["AutoModelForCausalLM"].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")
        model = stack["PeftModel"].from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def make_prompt(tokenizer: Any, trajectory: str) -> str:
    messages = [
        {"role": "system", "content": BINARY_SYSTEM_PROMPT},
        {"role": "user", "content": trajectory},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def infer_one(stack: dict[str, Any], tokenizer: Any, model: Any, trajectory: str) -> dict[str, Any]:
    torch = stack["torch"]
    prompt = make_prompt(tokenizer, trajectory)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_tokens = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
            "do_sample": GENERATION_CONFIG["do_sample"],
            "pad_token_id": tokenizer.eos_token_id,
        }
        if GENERATION_CONFIG["do_sample"]:
            generate_kwargs["temperature"] = GENERATION_CONFIG["temperature"]
        generated = model.generate(
            **inputs,
            **generate_kwargs,
        )
    output_ids = generated[0][input_tokens:]
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    output_tokens = int(output_ids.shape[0])
    parsed = parse_model_output(raw_output)
    return {
        "raw_output": raw_output,
        "pred": parsed.pred,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "strict_json": parsed.strict_json,
        "invalid_output": parsed.invalid_output,
        "parse_method": parsed.parse_method,
    }


def evaluate_dataset(
    stack: dict[str, Any],
    tokenizer: Any,
    model: Any,
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    method: str,
    limit: int | None,
) -> dict[str, Any]:
    rows = read_json(dataset_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list dataset at {dataset_path}")
    if limit is not None:
        rows = rows[:limit]

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, example in enumerate(rows):
        uid = str(example.get("uid", example.get("id", idx)))
        gold = normalize_gold_label(example["label"])
        trajectory = format_eval_trajectory(example, dataset_name)
        pred_info = infer_one(stack, tokenizer, model, trajectory)
        row = {
            "uid": uid,
            "dataset": dataset_name,
            "gold": gold,
            **pred_info,
        }
        row["is_correct"] = row["gold"] == row["pred"]
        predictions.append(row)
        if not row["is_correct"] or row["invalid_output"]:
            errors.append({"method": method, **row})
        if (idx + 1) % 25 == 0:
            print(f"[eval] {method}/{dataset_name}: {idx + 1}/{len(rows)}")

    write_jsonl(output_dir / f"predictions_{dataset_name}.jsonl", predictions)
    if errors:
        write_jsonl(Path("outputs/error_cases") / f"{method}_{dataset_name}.jsonl", errors)
    return compute_binary_metrics(predictions)


def evaluate_method(config: dict[str, Any], method: str, limit: int | None) -> dict[str, Any]:
    methods = config["methods"]
    if method not in methods:
        raise KeyError(f"Unknown method {method!r}. Available: {', '.join(methods)}")
    spec = methods[method]
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = import_eval_stack()
    tokenizer, model = load_model(stack, spec["model_path"], spec.get("adapter_path"))
    dataset_metrics = {}
    for dataset_name, path in config["datasets"].items():
        dataset_metrics[dataset_name] = evaluate_dataset(
            stack=stack,
            tokenizer=tokenizer,
            model=model,
            dataset_name=dataset_name,
            dataset_path=Path(path),
            output_dir=output_dir,
            method=method,
            limit=limit,
        )

    summary = {
        "method": method,
        "model_path": spec["model_path"],
        "adapter_path": spec.get("adapter_path"),
        "generation": GENERATION_CONFIG,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_methods.yaml")
    parser.add_argument("--method", default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    methods = list(config["methods"]) if args.method == "all" else [args.method]
    for method in methods:
        evaluate_method(config, method, args.limit)


if __name__ == "__main__":
    main()
