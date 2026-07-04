from types import SimpleNamespace

from agentdog_lite.loss_masking import IGNORE_INDEX, encode_assistant_only


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = ""
        for message in messages:
            role = message["role"]
            text += f"<{role}>{message['content']}</{role}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[ord(char) for char in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(idx) for idx in ids)


def test_system_user_labels_are_ignored_and_assistant_json_is_trained():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "classify"},
        {"role": "user", "content": "trajectory"},
        {"role": "assistant", "content": '{"judgment":"safe"}'},
    ]
    encoded = encode_assistant_only(tokenizer, messages, max_seq_len=256)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)

    assert all(label == IGNORE_INDEX for label in encoded.labels[:prompt_len])
    assert all(label != IGNORE_INDEX for label in encoded.labels[prompt_len:])
    assert tokenizer.decode(encoded.labels[prompt_len:]).startswith('{"judgment":"safe"}')


def test_long_prompt_truncates_without_truncating_assistant_target():
    tokenizer = FakeTokenizer()
    target = '{"judgment":"unsafe"}'
    messages = [
        {"role": "system", "content": "classify"},
        {"role": "user", "content": "x" * 500},
        {"role": "assistant", "content": target},
    ]
    encoded = encode_assistant_only(tokenizer, messages, max_seq_len=120)
    trained_ids = [idx for idx, label in zip(encoded.input_ids, encoded.labels, strict=True) if label != -100]

    assert encoded.truncated is True
    assert tokenizer.decode(trained_ids).startswith(target)
