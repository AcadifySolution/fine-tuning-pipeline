from data.preprocess import convert_to_messages_format


def test_messages_schema_is_preserved():
    sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    assert convert_to_messages_format(sample)["messages"] == sample["messages"]


def test_instruction_fallback():
    result = convert_to_messages_format({"instruction": "Question", "output": "Answer"})
    assert result["messages"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_invalid_messages_are_removed():
    result = convert_to_messages_format(
        {"messages": [{"role": "developer", "content": "ignored"}]}
    )
    assert result["messages"] == []
