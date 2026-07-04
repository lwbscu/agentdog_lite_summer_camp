"""Strict-first output parser for final AgentDoG-Lite predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_UNSAFE_RE = re.compile(r"\bunsafe\b", re.IGNORECASE)
_SAFE_RE = re.compile(r"\bsafe\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    pred: str
    strict_json: bool
    invalid_output: bool
    parse_method: str


def _valid_judgment(value: Any) -> str | None:
    if isinstance(value, str) and value.strip().lower() in {"safe", "unsafe"}:
        return value.strip().lower()
    return None


def _parse_json_object(text: str) -> tuple[str | None, bool]:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        judgment = _valid_judgment(obj.get("judgment"))
        if judgment is not None:
            strict = set(obj.keys()) == {"judgment"}
            return judgment, strict

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        judgment = _valid_judgment(obj.get("judgment"))
        if judgment is None:
            continue
        trailing = text[match.start() + end :].strip()
        leading = text[: match.start()].strip()
        strict = not leading and not trailing and set(obj.keys()) == {"judgment"}
        return judgment, strict
    return None, False


def parse_model_output(text: str) -> ParseResult:
    """Parse model output.

    Order is intentional: JSON first, then unsafe text, then safe text. Matching
    "safe" first would misparse "unsafe".
    """

    judgment, strict = _parse_json_object(text)
    if judgment is not None:
        return ParseResult(
            pred=judgment,
            strict_json=strict,
            invalid_output=not strict,
            parse_method="strict_json" if strict else "json_object",
        )

    if _UNSAFE_RE.search(text):
        return ParseResult(
            pred="unsafe",
            strict_json=False,
            invalid_output=True,
            parse_method="fallback_regex_unsafe",
        )
    if _SAFE_RE.search(text):
        return ParseResult(
            pred="safe",
            strict_json=False,
            invalid_output=True,
            parse_method="fallback_regex_safe",
        )
    return ParseResult(
        pred="unsafe",
        strict_json=False,
        invalid_output=True,
        parse_method="default_unsafe_unparseable",
    )

