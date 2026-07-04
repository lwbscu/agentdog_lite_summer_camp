from types import SimpleNamespace

from agentdog_lite.loss_masking import IGNORE_INDEX
from scripts.train_full_sft import (
    alpaca_row_to_messages,
    balanced_subset,
    encode_assistant_only_no_truncation,
    retained_checkpoint_ids,
    stratified_train_dev_split,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = ""
        for message in messages:
            text += f"<{message['role']}>{message['content']}</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[ord(char) for char in text])


def test_alpaca_row_extracts_trajectory_and_compacts_target():
    row = {
        "instruction": "prefix <BEGIN TRAJECTORY>\nturns\n<END TRAJECTORY> suffix",
        "input": "",
        "output": '{"judgment": "safe"}',
    }
    converted = alpaca_row_to_messages(row, source_index=7, extract_trajectory=True)

    assert converted["source_index"] == 7
    assert converted["label"] == "safe"
    assert converted["messages"][1]["content"] == "turns"
    assert converted["messages"][2]["content"] == '{"judgment":"safe"}'


def test_full_sft_labels_only_train_assistant_json():
    tokenizer = FakeTokenizer()
    row = {
        "source_index": 0,
        "label": "unsafe",
        "messages": [
            {"role": "system", "content": "classify"},
            {"role": "user", "content": "trajectory"},
            {"role": "assistant", "content": '{"judgment":"unsafe"}'},
        ],
    }
    encoded = encode_assistant_only_no_truncation(tokenizer, row, max_seq_len=256)
    first_label_idx = next(idx for idx, label in enumerate(encoded.labels) if label != IGNORE_INDEX)

    assert all(label == IGNORE_INDEX for label in encoded.labels[:first_label_idx])
    assert encoded.labels[first_label_idx:] == encoded.input_ids[first_label_idx:]


def test_stratified_split_and_balanced_subset_keep_labels():
    rows = [
        {"source_index": idx, "label": "safe" if idx < 10 else "unsafe", "messages": []}
        for idx in range(20)
    ]
    subset = balanced_subset(rows, max_samples=8, seed=1)
    train_rows, dev_rows = stratified_train_dev_split(subset, dev_ratio=0.25, seed=1)

    assert sum(row["label"] == "safe" for row in subset) == 4
    assert sum(row["label"] == "unsafe" for row in subset) == 4
    assert {row["label"] for row in train_rows} == {"safe", "unsafe"}
    assert {row["label"] for row in dev_rows} == {"safe", "unsafe"}


def test_retention_keeps_best_and_latest():
    entries = [
        {
            "checkpoint_id": "a",
            "eval_succeeded": True,
            "overall_accuracy": 0.7,
            "mean_macro_f1": 0.6,
            "step": 1,
            "created_at": "2026-01-01T00:00:01",
            "checkpoint_dir": "/tmp/a",
        },
        {
            "checkpoint_id": "b",
            "eval_succeeded": True,
            "overall_accuracy": 0.9,
            "mean_macro_f1": 0.7,
            "step": 2,
            "created_at": "2026-01-01T00:00:02",
            "checkpoint_dir": "/tmp/b",
        },
        {
            "checkpoint_id": "c",
            "eval_succeeded": True,
            "overall_accuracy": 0.8,
            "mean_macro_f1": 0.9,
            "step": 3,
            "created_at": "2026-01-01T00:00:03",
            "checkpoint_dir": "/tmp/c",
        },
    ]

    assert retained_checkpoint_ids(entries) == {"b", "c"}
