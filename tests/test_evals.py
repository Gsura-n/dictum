from dictum.evals import Case, check, load_cases, similarity


def test_cases_load_and_are_mostly_hard():
    cases = load_cases()
    assert len(cases) >= 25
    assert len({c.id for c in cases}) == len(cases), "case ids must be unique"
    hard = sum(c.difficulty == "hard" for c in cases)
    assert hard / len(cases) > 0.5


def test_check_include_exclude_regex():
    c = Case(id="x", mode="dictation", difficulty="easy", input="", reference="send it to Priya",
             checks={"include": ["priya"], "exclude": ["Sarah"], "regex": ["^send"], "max_len_ratio": 1.5})
    assert check(c, "Send it to Priya.") == []
    assert "contains 'Sarah'" in check(c, "Send it to Sarah and Priya.")
    assert any("too long" in f for f in check(c, "Send it to Priya right now please and thanks a lot"))


def test_exclude_is_word_bounded():
    c = Case(id="x", mode="dictation", difficulty="easy", input="", checks={"exclude": ["um"]})
    assert check(c, "The umbrella is here.") == []
    assert check(c, "Um, the umbrella.") != []


def test_similarity():
    assert similarity("a b c", "a b c") == 1.0
    assert similarity("", "a") == 0.0
