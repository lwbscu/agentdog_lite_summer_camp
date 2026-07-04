import pytest

from agentdog_lite.trajectory import (
    extract_trajectory_from_instruction,
    get_trajectory_extraction_stats,
    reset_trajectory_extraction_stats,
)


def test_marker_extracts_middle_content():
    instruction = "prefix <BEGIN TRAJECTORY>\nturns here\n<END TRAJECTORY> suffix"
    assert extract_trajectory_from_instruction(instruction) == "turns here"


def test_missing_marker_falls_back_to_full_instruction():
    reset_trajectory_extraction_stats()
    instruction = "full instruction without markers"
    assert extract_trajectory_from_instruction(instruction) == instruction
    assert get_trajectory_extraction_stats()["fallback_marker_missing_count"] == 1


def test_empty_instruction_raises():
    with pytest.raises(RuntimeError):
        extract_trajectory_from_instruction("")
