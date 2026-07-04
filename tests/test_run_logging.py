from pathlib import Path

from agentdog_lite import run_logging


def test_timestamped_log_dir_uses_liwenbo_prefix_without_creating(monkeypatch):
    monkeypatch.setattr(run_logging, "LOG_ROOT", Path("/tmp/agentdog_logs"))

    path = run_logging.make_timestamped_log_dir(
        "only_eval",
        "eval_demo",
        timestamp="20260705_010203",
        create=False,
    )

    assert path == Path("/tmp/agentdog_logs/only_eval/李文博_eval_demo_20260705_010203")
