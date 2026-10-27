import pytest
from dictum import dictionary as d

def test_parse_empty_and_none():
    assert d.parse(None) == []
    assert d.parse([]) == []

def test_parse_valid_string_and_dict():
    raw = ["apple", {"word": "banana", "sounds_like": ["nana"]}, {"word": "cherry"}]
    entries = d.parse(raw)
    assert len(entries) == 3
    assert entries[0].word == "apple"
    assert entries[0].sounds_like == []

    assert entries[1].word == "banana"
    assert entries[1].sounds_like == ["nana"]

    assert entries[2].word == "cherry"
    assert entries[2].sounds_like == []

def test_parse_invalid_entry_raises_value_error():
    with pytest.raises(ValueError, match="bad dictionary entry: {'not_word': 'test'}"):
        d.parse([{"not_word": "test"}])

    with pytest.raises(ValueError, match="bad dictionary entry: 123"):
        d.parse([123])
