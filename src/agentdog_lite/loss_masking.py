"""Assistant-only loss masking helpers for AgentDoG SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IGNORE_INDEX = -100
TRUNCATION_MARKER = "[TRUNCATED_MIDDLE_FOR_CONTEXT_WINDOW]"


@dataclass(frozen=True)
class AssistantOnlyEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    original_input_tokens: int
    prompt_tokens: int
    assistant_tokens: int
    truncated: bool


def _input_ids(encoded: Any) -> list[int]:
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return _input_ids(tokenizer(text, add_special_tokens=False))


def render_prompt_and_full(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, str]:
    if len(messages) < 3:
        raise RuntimeError("Expected messages=[system,user,assistant] for SFT.")
    if messages[-1].get("role") != "assistant":
        raise RuntimeError("Last SFT message must be the assistant target.")
    assistant_content = str(messages[-1].get("content", ""))
    if not assistant_content.strip():
        raise RuntimeError("Assistant target is empty; refusing to create silent zero-loss row.")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def encode_assistant_only(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_seq_len: int,
) -> AssistantOnlyEncoding:
    """Tokenize a chat row so only assistant target tokens contribute loss."""

    if max_seq_len <= 0:
        raise RuntimeError(f"max_seq_len must be positive, got {max_seq_len}")

    prompt_text, full_text = render_prompt_and_full(tokenizer, messages)
    prompt_ids = encode_text(tokenizer, prompt_text)
    full_ids = encode_text(tokenizer, full_text)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Chat template full text is not prefixed by the generation prompt; "
            "cannot safely mask assistant-only loss."
        )

    assistant_ids = full_ids[len(prompt_ids) :]
    if not assistant_ids:
        raise RuntimeError("Assistant target tokenization produced zero tokens.")
    if len(assistant_ids) >= max_seq_len:
        raise RuntimeError(
            "Assistant target is longer than max_seq_len and would be truncated: "
            f"assistant_tokens={len(assistant_ids)}, max_seq_len={max_seq_len}"
        )

    original_input_tokens = len(full_ids)
    truncated = False
    masked_prompt_ids = prompt_ids
    if len(full_ids) > max_seq_len:
        prompt_budget = max_seq_len - len(assistant_ids)
        marker_ids = encode_text(tokenizer, TRUNCATION_MARKER)
        if prompt_budget <= len(marker_ids) + 2:
            raise RuntimeError(
                "Not enough context budget to preserve assistant target plus truncation marker: "
                f"prompt_budget={prompt_budget}, marker_tokens={len(marker_ids)}"
            )
        keep_budget = prompt_budget - len(marker_ids)
        head_tokens = max(1, int(keep_budget * 0.30))
        tail_tokens = max(1, keep_budget - head_tokens)
        if head_tokens + tail_tokens > keep_budget:
            tail_tokens = keep_budget - head_tokens
        masked_prompt_ids = prompt_ids[:head_tokens] + marker_ids + prompt_ids[-tail_tokens:]
        truncated = True

    input_ids = masked_prompt_ids + assistant_ids
    labels = [IGNORE_INDEX] * len(masked_prompt_ids) + assistant_ids
    if len(input_ids) > max_seq_len:
        raise RuntimeError(
            f"Internal truncation bug: encoded length {len(input_ids)} exceeds {max_seq_len}"
        )
    if labels[-len(assistant_ids) :] != assistant_ids:
        raise RuntimeError("Assistant target labels were not preserved after truncation.")

    return AssistantOnlyEncoding(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        original_input_tokens=original_input_tokens,
        prompt_tokens=len(prompt_ids),
        assistant_tokens=len(assistant_ids),
        truncated=truncated,
    )


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return int(round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight))


def summarize_encodings(encodings: list[AssistantOnlyEncoding]) -> dict[str, Any]:
    lengths = sorted(encoding.original_input_tokens for encoding in encodings)
    total = len(lengths)
    return {
        "num_samples": total,
        "input_tokens": {
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": lengths[-1] if lengths else 0,
        },
        "over_8192_ratio": sum(length > 8192 for length in lengths) / total if total else 0.0,
        "over_16384_ratio": sum(length > 16384 for length in lengths) / total if total else 0.0,
        "truncated_count": sum(encoding.truncated for encoding in encodings),
    }


def build_token_length_summary(
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    max_seq_len: int,
) -> dict[str, Any]:
    train_encodings = [
        encode_assistant_only(tokenizer, row["messages"], max_seq_len=max_seq_len)
        for row in train_rows
    ]
    dev_encodings = [
        encode_assistant_only(tokenizer, row["messages"], max_seq_len=max_seq_len)
        for row in dev_rows
    ]
    return {
        "max_seq_len": max_seq_len,
        "train": summarize_encodings(train_encodings),
        "dev": summarize_encodings(dev_encodings),
    }
