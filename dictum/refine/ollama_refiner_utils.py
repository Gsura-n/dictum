"""Prompt building and output cleanup shared by LLM refiners."""
from __future__ import annotations

import re

from ..config import Mode

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)
_TAGS = re.compile(r"</?transcript>")


def clean_output(text: str) -> str:
    text = _THINK.sub("", text)
    text = _FENCE.sub("", text)
    text = _TAGS.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def wrap(text: str) -> str:
    return f"<transcript>{text}</transcript>"


def build_messages(mode: Mode, user_text: str, dictionary: list[str]) -> list[dict]:
    system = mode.prompt.strip()
    if not mode.dictionary_hint:
        dictionary = []
    for k, v in (mode.vars or {}).items():
        system = system.replace("{" + k + "}", str(v))
    if dictionary:
        system += ("\n\nThese names and terms are spelled exactly like this when they occur: "
                   + ", ".join(dictionary) + ". Do not use them anywhere they were not said.")
    msgs = [{"role": "system", "content": system}]
    for ex in mode.examples:
        msgs.append({"role": "user", "content": wrap(ex["in"])})
        msgs.append({"role": "assistant", "content": ex["out"]})
    msgs.append({"role": "user", "content": wrap(user_text)})
    return msgs


