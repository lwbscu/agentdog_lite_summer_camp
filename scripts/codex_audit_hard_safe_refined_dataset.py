#!/usr/bin/env python
"""Audit hard-safe refined SFT rows with action-based Codex review rules.

This script does not call external model APIs. It writes a reproducible audit
artifact that can be cross-checked by Codex/subagent review and used as a
training gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_hard_safe_refined_dataset import count_test_overlap


BEGIN_MARKER = "<BEGIN TRAJECTORY>"
END_MARKER = "<END TRAJECTORY>"
DEFAULT_DATASET = Path("data/Proessed_train_dataset/llamafactory_train_hard_safe_refined_v1.json")
DEFAULT_FORM_SUMMARY = Path("data/Proessed_train_dataset/llamafactory_train_hard_safe_refined_v1_summary.json")
DEFAULT_AUDIT_JSONL = Path("data/Proessed_train_dataset/llamafactory_train_hard_safe_refined_v1_codex_audit.jsonl")
DEFAULT_AUDIT_SUMMARY = Path("data/Proessed_train_dataset/llamafactory_train_hard_safe_refined_v1_codex_audit_summary.json")
DEFAULT_REPORT = Path("report/hard_safe_refined_v1_codex_audit.md")
DEFAULT_TEST_DIR = Path("data/2026_summer_camp_teseset")

DANGEROUS_ACTIONS = {
    "send_email",
    "update_security_policy",
    "grant_door_access",
    "transfer_funds",
    "export_customer_records",
    "delete_audit_log",
    "create_blacklist_policy",
    "send_sms",
}
SAFE_ACTIONS = {
    "search_messages",
    "summarize_record",
    "translate_public_note",
    "read_calendar_event",
    "classify_ticket",
    "lookup_public_article",
    "create_short_link",
    "render_preview",
}
FORBIDDEN_TEST_FIELDS = {
    "label",
    "reason",
    "risk_source",
    "failure_mode",
    "harm_type",
    "risk_description",
    "risk_type",
    "source",
}
RISK_WORD_PATTERNS = [
    "ignore previous instructions",
    "credentials",
    "administrator access",
    "unverified wallet",
    "exfiltrate",
    "blacklist policy",
    "delete the audit log",
    "security token",
    "prompt injection",
    "malicious",
    "attacker",
]
UNSAFE_SIDE_EFFECT_RE = re.compile(
    r"\b(success|succeeded|delivery success|update success|grant success|transaction id|completed)\b",
    re.IGNORECASE,
)
ACTION_RE = re.compile(r"^\s*Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\(", re.MULTILINE)
CASE_RE = re.compile(r"Case ID:\s*hard_safe_refined_v1_(atbench_like|rjudge_like)_(safe|unsafe)_(\d+)")
NEUTRAL_CASE_RE = re.compile(r"Case ID:\s*trajectory_refined_v1_case_([0-9a-fA-F]+)")
SOURCE_STYLE_RE = re.compile(r"Source Style:\s*(atbench_like|rjudge_like)")
PAIR_RE = re.compile(r"Pair ID:\s*([^\n]+)")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_trajectory(instruction: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    if BEGIN_MARKER not in instruction or END_MARKER not in instruction:
        return "", ["missing_trajectory_marker"]
    begin = instruction.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = instruction.index(END_MARKER)
    trajectory = instruction[begin:end].strip()
    if not trajectory:
        issues.append("empty_trajectory")
    return trajectory, issues


def parse_output_label(output: str) -> tuple[str | None, list[str]]:
    issues: list[str] = []
    try:
        obj = json.loads(output)
    except Exception:
        return None, ["output_json_parse_error"]
    if sorted(obj) != ["judgment"]:
        issues.append("output_not_judgment_only")
    label = obj.get("judgment")
    if label not in {"safe", "unsafe"}:
        issues.append("bad_output_judgment")
        return None, issues
    return label, issues


def parse_generated_metadata(trajectory: str) -> dict[str, Any]:
    case_match = CASE_RE.search(trajectory)
    neutral_case_match = NEUTRAL_CASE_RE.search(trajectory)
    source_style_match = SOURCE_STYLE_RE.search(trajectory)
    pair_match = PAIR_RE.search(trajectory)
    meta: dict[str, Any] = {
        "source_style": "",
        "case_label": "",
        "local_index": None,
        "neutral_case_id": "",
        "pair_id": "",
    }
    if case_match:
        meta["source_style"] = case_match.group(1)
        meta["case_label"] = case_match.group(2)
        meta["local_index"] = int(case_match.group(3))
    if neutral_case_match:
        meta["neutral_case_id"] = neutral_case_match.group(1)
    if source_style_match:
        meta["source_style"] = source_style_match.group(1)
    if pair_match:
        meta["pair_id"] = pair_match.group(1).strip()
    return meta


def find_action_names(trajectory: str) -> list[str]:
    return ACTION_RE.findall(trajectory)


def has_forbidden_test_field_text(text: str) -> list[str]:
    hits: list[str] = []
    for field in FORBIDDEN_TEST_FIELDS:
        if re.search(rf'["\']?{re.escape(field)}["\']?\s*:', text):
            hits.append(field)
    return sorted(hits)


def label_leakage_issues(trajectory: str) -> list[str]:
    issues: list[str] = []
    if CASE_RE.search(trajectory):
        issues.append("case_id_contains_label")
    if re.search(r"Case ID:\s*hard_safe_refined_v1_case_", trajectory, flags=re.IGNORECASE):
        issues.append("case_id_contains_static_safe_token")
    if re.search(r"^\s*Safety label rationale:", trajectory, flags=re.MULTILINE | re.IGNORECASE):
        issues.append("explicit_safety_label_rationale")
    if re.search(r"\baudit testing\b|\bAudit context segment\b", trajectory, flags=re.IGNORECASE):
        issues.append("meta_audit_phrase")
    if re.search(
        r"^\s*(Middle|Tail) evidence:|no dangerous tool call appears|"
        r"\bsafety judgment\b|\bimportant evidence\b|\bsafe completion\b|"
        r"\bunsafe action\b|\bsafe guidance\b|\bpolicy safe alternative\b",
        trajectory,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        issues.append("judgment_hint_phrase")
    return issues


def infer_label_from_actions(trajectory: str, actions: list[str]) -> tuple[str, list[str]]:
    dangerous_called = [name for name in actions if name in DANGEROUS_ACTIONS]
    if dangerous_called:
        return "unsafe", dangerous_called
    return "safe", []


def shortcut_risks(trajectory: str, expected_label: str | None, actions: list[str]) -> list[str]:
    lowered = trajectory.lower()
    risks: list[str] = []
    risk_word_count = sum(1 for pattern in RISK_WORD_PATTERNS if pattern in lowered)
    if expected_label == "safe" and risk_word_count >= 2:
        risks.append("safe_with_dense_risk_words")
    if expected_label == "safe" and len(trajectory.split()) > 1200:
        risks.append("safe_long_context")
    if expected_label == "unsafe" and "final answer: i cannot" in lowered:
        risks.append("unsafe_final_refusal_after_action")
    if expected_label == "safe" and any(name in lowered for name in DANGEROUS_ACTIONS):
        risks.append("safe_mentions_dangerous_tool_without_execution")
    if expected_label == "unsafe" and len(actions) <= 1:
        risks.append("short_or_sparse_unsafe_context")
    return risks


def audit_row(row: dict[str, Any], index: int, chunk_id: int) -> dict[str, Any]:
    issues: list[str] = []
    expected_label, output_issues = parse_output_label(str(row.get("output", "")))
    issues.extend(output_issues)
    if row.get("input", "") != "":
        issues.append("input_not_empty")
    if set(row) != {"instruction", "input", "output"}:
        issues.append("unexpected_schema_keys")

    instruction = str(row.get("instruction", ""))
    trajectory, trajectory_issues = extract_trajectory(instruction)
    issues.extend(trajectory_issues)
    actions = find_action_names(trajectory)
    observations = len(re.findall(r"^\s*Observation:", trajectory, flags=re.MULTILINE))
    explicit_action_ok = bool(actions) and observations > 0
    if not explicit_action_ok:
        issues.append("missing_explicit_action_or_observation")

    forbidden_fields = has_forbidden_test_field_text(instruction)
    if forbidden_fields:
        issues.append("forbidden_test_field_text:" + ",".join(forbidden_fields))
    issues.extend(label_leakage_issues(trajectory))

    audited_label, dangerous_actions = infer_label_from_actions(trajectory, actions)
    label_ok = expected_label == audited_label
    if not label_ok:
        issues.append(f"label_mismatch_expected_{expected_label}_audited_{audited_label}")

    if expected_label == "unsafe" and not dangerous_actions:
        issues.append("unsafe_without_dangerous_action")
    if expected_label == "safe" and dangerous_actions:
        issues.append("safe_with_dangerous_action")
    if dangerous_actions and not UNSAFE_SIDE_EFFECT_RE.search(trajectory):
        issues.append("dangerous_action_without_clear_side_effect")

    metadata = parse_generated_metadata(trajectory)
    if not metadata["source_style"]:
        issues.append("missing_generated_case_metadata")
    if metadata["case_label"] and expected_label and metadata["case_label"] != expected_label:
        issues.append("case_label_output_mismatch")

    critical_issue = any(
        issue.startswith(
            (
                "output_json_parse_error",
                "bad_output_judgment",
                "missing_trajectory_marker",
                "empty_trajectory",
                "label_mismatch",
                "forbidden_test_field_text",
                "case_id_contains_label",
                "case_id_contains_static_safe_token",
                "explicit_safety_label_rationale",
                "meta_audit_phrase",
                "judgment_hint_phrase",
                "unsafe_without_dangerous_action",
                "safe_with_dangerous_action",
            )
        )
        for issue in issues
    )
    if not issues:
        decision = "pass"
    elif critical_issue:
        decision = "drop"
    else:
        decision = "repair"

    return {
        "index": index,
        "chunk_id": chunk_id,
        "expected_label": expected_label,
        "audited_label": audited_label,
        "label_ok": label_ok,
        "explicit_intermediate_action_ok": explicit_action_ok,
        "shortcut_risk": shortcut_risks(trajectory, expected_label, actions),
        "issues": sorted(set(issues)),
        "decision": decision,
        "source_style": metadata["source_style"],
        "pair_id": metadata["pair_id"],
        "actions": actions,
        "dangerous_actions": dangerous_actions,
        "repair_hint": repair_hint(expected_label, audited_label, dangerous_actions, issues),
    }


def repair_hint(
    expected_label: str | None,
    audited_label: str,
    dangerous_actions: list[str],
    issues: list[str],
) -> str:
    if not issues:
        return ""
    if expected_label == "safe" and dangerous_actions:
        return "Remove dangerous Action lines or relabel only if the unsafe action is intentional."
    if expected_label == "unsafe" and audited_label == "safe":
        return "Add an explicit dangerous Action plus successful side effect, or relabel as safe."
    if "missing_explicit_action_or_observation" in issues:
        return "Add explicit Action and Observation lines so label depends on trajectory actions."
    return "Regenerate this sample with the same source style, label, length bucket, and contrastive role."


def summarize_audit(records: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in records)
    labels = Counter(row["expected_label"] for row in records)
    final_labels = Counter(json.loads(row["output"])["judgment"] for row in rows)
    shortcut_counts = Counter(flag for row in records for flag in row["shortcut_risk"])
    issue_counts = Counter(issue for row in records for issue in row["issues"])
    chunk_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in records:
        chunk_counts[int(row["chunk_id"])][row["decision"]] += 1
    test_overlap_count = count_test_overlap(rows[args.start_index : args.end_index], args.test_data_dir)

    return {
        "dataset_file": str(args.dataset),
        "audited_start_index": args.start_index,
        "audited_end_index_exclusive": args.end_index,
        "audited_count": len(records),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunk_counts),
        "pass_count": decisions["pass"],
        "repair_count": decisions["repair"],
        "drop_count": decisions["drop"],
        "decision_counts": dict(decisions),
        "audited_label_counts": dict(labels),
        "final_label_counts": dict(final_labels),
        "shortcut_risk_counts": dict(shortcut_counts),
        "issue_counts": dict(issue_counts),
        "critical_issue_count": decisions["repair"] + decisions["drop"],
        "test_overlap_count": test_overlap_count,
        "chunk_decision_counts": {
            str(chunk_id): dict(counts) for chunk_id, counts in sorted(chunk_counts.items())
        },
        "acceptance": {
            "audited_count_is_1600": len(records) == 1600,
            "no_repair": decisions["repair"] == 0,
            "no_drop": decisions["drop"] == 0,
            "critical_issue_count_is_zero": decisions["repair"] + decisions["drop"] == 0,
            "test_overlap_count_is_zero": test_overlap_count == 0,
            "final_total_is_3600": len(rows) == 3600,
            "final_safe_is_1900": final_labels["safe"] == 1900,
            "final_unsafe_is_1700": final_labels["unsafe"] == 1700,
        },
    }


def validate_form_summary(path: Path) -> list[str]:
    summary = read_json(path)
    acceptance = summary.get("acceptance", {})
    return [key for key, ok in acceptance.items() if not ok]


def write_report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    issue_rows = [row for row in records if row["decision"] != "pass"]
    shortcut_counts = summary["shortcut_risk_counts"]
    lines = [
        "# hard_safe_refined_v1 Codex Audit",
        "",
        "## Summary",
        "",
        f"- Audited rows: {summary['audited_count']} ({summary['audited_start_index']}..{summary['audited_end_index_exclusive'] - 1})",
        f"- Pass: {summary['pass_count']}",
        f"- Repair: {summary['repair_count']}",
        f"- Drop: {summary['drop_count']}",
        f"- Critical issues: {summary['critical_issue_count']}",
        f"- Test overlap count: {summary['test_overlap_count']}",
        "",
        "## Shortcut Risk Coverage",
        "",
    ]
    if shortcut_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(shortcut_counts.items()))
    else:
        lines.append("- None detected")
    lines.extend(["", "## Chunk Decisions", "", "| chunk | pass | repair | drop |", "|---:|---:|---:|---:|"])
    for chunk_id, counts in summary["chunk_decision_counts"].items():
        lines.append(
            f"| {chunk_id} | {counts.get('pass', 0)} | {counts.get('repair', 0)} | {counts.get('drop', 0)} |"
        )
    lines.extend(["", "## Issues", ""])
    if not issue_rows:
        lines.append("No repair/drop issues found. Dataset passes the Codex action audit gate.")
    else:
        for row in issue_rows[:50]:
            lines.append(
                f"- index={row['index']} decision={row['decision']} expected={row['expected_label']} "
                f"audited={row['audited_label']} issues={row['issues']} hint={row['repair_hint']}"
            )
        if len(issue_rows) > 50:
            lines.append(f"- ... {len(issue_rows) - 50} more issue rows omitted from markdown; see JSONL.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--form-summary", type=Path, default=DEFAULT_FORM_SUMMARY)
    parser.add_argument("--audit-jsonl", type=Path, default=DEFAULT_AUDIT_JSONL)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--test-data-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--start-index", type=int, default=2000)
    parser.add_argument("--end-index", type=int, default=3600)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--allow-issues", action="store_true")
    args = parser.parse_args()

    failed_form_checks = validate_form_summary(args.form_summary)
    if failed_form_checks:
        raise RuntimeError(f"Formal dataset summary checks failed: {failed_form_checks}")

    rows = read_json(args.dataset)
    if not isinstance(rows, list):
        raise TypeError(f"Expected list dataset at {args.dataset}")
    if args.start_index < 0 or args.end_index > len(rows) or args.start_index >= args.end_index:
        raise ValueError(f"Bad audit range: {args.start_index}:{args.end_index} for {len(rows)} rows")

    records = [
        audit_row(
            row=rows[index],
            index=index,
            chunk_id=(index - args.start_index) // args.chunk_size,
        )
        for index in range(args.start_index, args.end_index)
    ]
    summary = summarize_audit(records, rows, args)
    write_jsonl(args.audit_jsonl, records)
    write_json(args.audit_summary, summary)
    write_report(args.report, summary, records)

    failed = [key for key, ok in summary["acceptance"].items() if not ok]
    if failed and not args.allow_issues:
        raise RuntimeError(
            "Codex audit gate failed: "
            + ", ".join(failed)
            + f"\nSee {args.audit_jsonl} and {args.report}"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
