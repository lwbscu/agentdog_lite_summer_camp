"""Trajectory formatting helpers.

Only trajectory/tool/profile fields are rendered for evaluation. Gold labels,
reasons, risk annotations, and other forbidden fields are intentionally excluded.
"""

from __future__ import annotations

import json
import re
from typing import Any


FORBIDDEN_EVAL_FIELDS = {
    "label",
    "reason",
    "risk_source",
    "failure_mode",
    "harm_type",
    "risk_description",
    "risk_type",
    "source",
}

BEGIN_MARKER = "<BEGIN TRAJECTORY>"
END_MARKER = "<END TRAJECTORY>"


def extract_trajectory_from_instruction(instruction: str) -> str:
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}\s*(.*?)\s*{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    match = pattern.search(instruction)
    if not match:
        raise ValueError("Could not find trajectory markers in training instruction.")
    return match.group(1).strip()


def normalize_gold_label(value: Any) -> str:
    if isinstance(value, bool):
        return "unsafe" if value else "safe"
    if isinstance(value, int):
        if value == 0:
            return "safe"
        if value == 1:
            return "unsafe"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "safe", "benign"}:
            return "safe"
        if lowered in {"1", "unsafe", "harmful"}:
            return "unsafe"
    raise ValueError(f"Cannot normalize label: {value!r}")


def _render_turn(turn: dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).upper()
    if role == "USER":
        return f"[USER]: {turn.get('content', '')}"
    if role in {"AGENT", "ASSISTANT"}:
        parts = []
        for key, value in turn.items():
            if key == "role" or value in (None, ""):
                continue
            parts.append(f"[{key.upper()}]: {stringify(value)}")
        return "[AGENT]:\n" + "\n".join(parts) if parts else "[AGENT]:"
    if role in {"ENVIRONMENT", "TOOL", "OBSERVATION"}:
        return f"[ENVIRONMENT]: {stringify(turn.get('content', turn))}"
    content = turn.get("content", {k: v for k, v in turn.items() if k != "role"})
    return f"[{role}]: {stringify(content)}"


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def format_contents(contents: Any) -> str:
    lines = ["=== Conversation History ==="]
    if not isinstance(contents, list):
        lines.append(stringify(contents))
        return "\n".join(lines)

    for round_data in contents:
        turns = round_data if isinstance(round_data, list) else [round_data]
        for turn in turns:
            if isinstance(turn, dict):
                lines.append("")
                lines.append(_render_turn(turn))
            else:
                lines.append("")
                lines.append(stringify(turn))
    return "\n".join(lines).strip()


def format_eval_trajectory(example: dict[str, Any], dataset: str) -> str:
    allowed = {k: v for k, v in example.items() if k not in FORBIDDEN_EVAL_FIELDS}
    lines: list[str] = []

    profile = allowed.get("profile")
    if profile:
        lines.append("=== Agent Profile ===")
        lines.append(stringify(profile))
        lines.append("")

    tools = allowed.get("tool_used") or allowed.get("tools") or allowed.get("available_tools")
    if tools:
        lines.append("=== Available Tools ===")
        lines.append(stringify(tools))
        lines.append("")

    scenario = allowed.get("scenario")
    if scenario:
        lines.append("=== Scenario ===")
        lines.append(stringify(scenario))
        lines.append("")

    if "contents" not in allowed:
        raise ValueError(f"Missing contents in {dataset} example id={allowed.get('id')!r}")
    lines.append(format_contents(allowed["contents"]))
    return "\n".join(lines).strip()

