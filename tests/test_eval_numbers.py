"""Synthetic, model-free checks for number equivalence in personal STT scoring."""
from itertools import product

import pytest

from dictum.evals import normalize_numbers, wer


@pytest.mark.parametrize("forms", [
    ("250", "two hundred fifty", "two hundred and fifty"),
    ("34.2", "thirty four point two", "thirty-four point two"),
    ("0.13", ".13", "point one three", "zero point one three"),
    ("p50", "p fifty", "P50"),
    ("4-bit", "four bit", "4 bit"),
    ("0", "zero"),
    ("1001", "one thousand one", "one thousand and one"),
    ("1250", "one thousand two hundred and fifty"),
    ("1000000", "one million"),
    ("1000000001", "one billion and one"),
    ("1000000000000", "one trillion"),
    ("12.003", "twelve point zero zero three"),
])
def test_required_equivalences_in_context(forms):
    canonical = normalize_numbers(f"Measured {forms[0]} today.")
    for form in forms:
        assert normalize_numbers(f"Measured {form} today.") == canonical
        assert wer(normalize_numbers(form), normalize_numbers(forms[0])) == 0


UNITS = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
# An independent finite oracle for the generated cardinal/decimal properties.
SMALL = UNITS + TEENS + [
    ten + (f" {unit}" if unit != "zero" else "") for ten, unit in product(TENS, UNITS)
]


def test_all_cardinals_below_one_thousand_and_optional_and():
    for n, words in enumerate(SMALL):
        assert normalize_numbers(str(n)) == normalize_numbers(words)
    for hundreds, rest in product(range(1, 10), range(100)):
        number = str(hundreds * 100 + rest)
        prefix = f"{UNITS[hundreds]} hundred"
        variants = [prefix] if not rest else [f"{prefix} {SMALL[rest]}", f"{prefix} and {SMALL[rest]}"]
        for words in variants:
            assert normalize_numbers(number) == normalize_numbers(words)


def test_fractional_digits_keep_their_order_and_trailing_zeroes():
    for integer, a, b in product((0, 4, 34, 99), range(10), range(10)):
        numeric = f"{integer}.{a}{b}"
        spoken = f"{SMALL[integer]} point {UNITS[a]} {UNITS[b]}"
        assert normalize_numbers(numeric) == normalize_numbers(spoken)
        assert normalize_numbers(numeric) != normalize_numbers(f"{integer}.{a}{b}0")


def test_number_boundaries_in_technical_phrases():
    for n, spoken in enumerate(SMALL):
        assert normalize_numbers(f"p{n}") == normalize_numbers(f"p {spoken}")
        assert normalize_numbers(f"{n}-bit") == normalize_numbers(f"{spoken} bit")


def test_idempotence_case_spacing_and_punctuation_properties():
    for phrase in ["", "plain words", "one and two", "zero zero point one", *SMALL,
                   *map(str, range(1000)), "p50", "4-bit", "0.13", "34.2"]:
        normal = normalize_numbers(phrase)
        assert normalize_numbers(normal) == normal
        assert normalize_numbers(f"  {phrase.upper()}!  ") == normal
        assert normalize_numbers(phrase.replace(" ", "  \n")) == normal


@pytest.mark.parametrize("text", ["one and two", "bread and butter", "don't change words", "sha256 x86"])
def test_unrelated_words_conjunctions_and_identifiers_are_preserved(text):
    assert normalize_numbers(text) == text


def test_year_style_is_explicitly_not_equivalent():
    assert normalize_numbers("2023") != normalize_numbers("twenty twenty three")
