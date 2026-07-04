from agentdog_lite.metrics import compute_binary_metrics


def test_metrics_include_token_cost_and_refusal_rates():
    rows = [
        {
            "gold": "safe",
            "pred": "unsafe",
            "strict_json": True,
            "invalid_output": False,
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "raw_output": '{"judgment":"unsafe"}',
        },
        {
            "gold": "unsafe",
            "pred": "unsafe",
            "strict_json": False,
            "invalid_output": True,
            "input_tokens": 20,
            "output_tokens": 4,
            "total_tokens": 24,
            "raw_output": "I cannot decide",
        },
    ]

    metrics = compute_binary_metrics(rows)

    assert metrics["total_input_tokens"] == 30
    assert metrics["total_output_tokens"] == 6
    assert metrics["total_tokens"] == 36
    assert metrics["token_cost_total_tokens"] == 36
    assert metrics["estimated_cost_usd"] == 0.0
    assert metrics["refusal_rate_safe_to_unsafe"] == 1.0
    assert metrics["raw_refusal_text_rate"] == 0.5
