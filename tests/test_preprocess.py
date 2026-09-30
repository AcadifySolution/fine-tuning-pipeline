from data.preprocess import convert_to_messages_format


def test_messages_schema_is_preserved():
    sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    result = convert_to_messages_format(sample)
    assert result["messages"] == sample["messages"]


def test_instruction_response_fallback():
    result = convert_to_messages_format({"instruction": "Question", "output": "Answer"})
    assert result["messages"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_empty_or_invalid_examples_are_rejected():
    assert convert_to_messages_format({"messages": [{"role": "system", "content": ""}]})["messages"] == []
