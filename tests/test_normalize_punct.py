"""Spoken punctuation: speech-to-text punctuation around a command, and talking ABOUT punctuation.

Spec: a punctuation command replaces the punctuation the speech-to-text put right before it and absorbs
any it put right after; a layout command (new line, new paragraph) leaves the preceding punctuation alone.
A trigger word used as a noun ("the Oxford comma", "a colon after the heading") is left as written, and
"literal" in front of a trigger forces the word through.
"""
import pytest

from dictum import normalize as n


def run(text):
    return n.join(n.apply(text))


@pytest.mark.parametrize("raw, want", [
    ("requests, comma, while larger models", "requests, while larger models"),
    ("reasoning, full stop.", "reasoning."),
    ("Does that work for you? Question mark.", "Does that work for you?"),
    ("the movie. Colon. New line. The visuals", "the movie:\nThe visuals"),
    ("Hi Sarah, comma, new line. Thanks", "Hi Sarah,\nThanks"),
    ("Best, comma, new line, Gauttam.", "Best,\nGauttam."),
    ("apples comma oranges comma and pears", "apples, oranges, and pears"),
])
def test_command_replaces_stt_punctuation_around_it(raw, want):
    assert run(raw) == want


@pytest.mark.parametrize("raw, want", [
    ("Hi Sarah, new line thanks", "Hi Sarah,\nthanks"),
    ("Thanks. New paragraph. I'm free", "Thanks.\n\nI'm free"),
    ("Hi Sarah, new paragraph thanks", "Hi Sarah,\n\nthanks"),
])
def test_layout_command_keeps_the_punctuation_before_it(raw, want):
    assert run(raw) == want


@pytest.mark.parametrize("text", [
    "I always use the Oxford comma in lists",
    "a comma splice joins two sentences",
    "the semicolon is optional",
    "add a colon after the heading",
    "the question mark at the end looks wrong",
    "an exclamation mark is too loud for a work email",
    "the bus came to a full stop at the light",
    "the new line of products launches Friday",
    "please add a new line character at the end of the file",
    "the next line of defense is the firewall",
    "the next paragraph of the essay needs a citation",
    "the dot com bubble burst in two thousand",
])
def test_trigger_used_as_a_noun_is_left_alone(text):
    assert run(text) == text


@pytest.mark.parametrize("raw, want", [
    ("Hi Sarah comma are you free Tuesday", "Hi Sarah, are you free Tuesday"),
    ("sorry comma was that you", "sorry, was that you"),
    ("okay comma is the build green", "okay, is the build green"),
    ("send it to the team comma then wait", "send it to the team, then wait"),
    ("thanks to the whole team comma", "thanks to the whole team,"),
    ("Monday comma after lunch works", "Monday, after lunch works"),
    ("that is the plan comma of course", "that is the plan, of course"),
    ("what is the status question mark", "what is the status?"),
    ("first the budget new line then the schedule", "first the budget\nthen the schedule"),
    ("no comma I disagree", "no, I disagree"),
])
def test_a_command_after_a_noun_phrase_still_fires(raw, want):
    assert run(raw) == want


@pytest.mark.parametrize("raw, want", [
    ("use a literal comma here", "use a comma here"),
    ("type literal period at the end", "type period at the end"),
    ("the literal new line is a command", "the new line is a command"),
    ("Literal question mark", "question mark"),
    ("apples literal comma oranges comma pears", "apples comma oranges, pears"),
])
def test_literal_forces_the_word_through(raw, want):
    assert run(raw) == want
