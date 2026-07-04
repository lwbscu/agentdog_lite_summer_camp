import pytest

from agentdog_lite.parser import parse_model_output


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"judgment":"safe"}', "safe"),
        ('{"judgment":"unsafe"}', "unsafe"),
        ("unsafe", "unsafe"),
        ("safe", "safe"),
        ("not safe", "unsafe"),
        ("This is unsafe.", "unsafe"),
        ("This is safe.", "safe"),
        ("不安全", "unsafe"),
        ("安全", "safe"),
        ("I cannot decide", "unsafe"),
    ],
)
def test_parse_outputs(text, expected):
    assert parse_model_output(text).pred == expected


def test_unsafe_is_not_misparsed_as_safe():
    result = parse_model_output("unsafe")
    assert result.pred == "unsafe"
    assert result.parse_method == "regex"


def test_reasoning_json_safe_is_valid_but_not_strict():
    result = parse_model_output('{"reasoning":"no unsafe action","judgment":"safe"}')

    assert result.pred == "safe"
    assert result.parse_method == "json"
    assert result.strict_json is False
    assert result.invalid_output is False


def test_reasoning_json_unsafe_is_valid_but_not_strict():
    result = parse_model_output('{"reasoning":"tool executed harm","judgment":"unsafe"}')

    assert result.pred == "unsafe"
    assert result.parse_method == "json"
    assert result.strict_json is False
    assert result.invalid_output is False
