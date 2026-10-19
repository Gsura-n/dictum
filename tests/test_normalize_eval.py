"""Tests for the normalize measurement scorer: attribution of misfires to the rule that fired, the
metrics, and the CLI. They do NOT assert that normalize gets cases right; the known misfires fail today
and the suite is measurement only. Fixtures are sentences that fire; the noun-use sentences that used to
fire are asserted not to in test_normalize_punct.py. Case-file checks are in test_normalize_cases.py."""
import pytest
import yaml
from typer.testing import CliRunner

from dictum import normalize_eval as ne


def test_loader_rejects_a_malformed_case(tmp_path, monkeypatch):
    bad = tmp_path / "dev.yaml"
    bad.write_text(yaml.safe_dump([{"id": "x", "rule": "nonsense", "fire": True, "input": "a", "expected": "b"}]))
    monkeypatch.setattr(ne, "CASES_DIR", tmp_path)
    with pytest.raises(ValueError, match="nonsense"):
        ne.load("dev")


@pytest.mark.parametrize("text, fired", [
    ("I had had enough of that", []),
    ("hello semicolon world", ["semicolon"]),
    ("really question mark that is surprising", ["question_mark"]),
    ("note colon the office closes early", ["colon"]),
    ("milk new line eggs", ["new_line"]),
    ("dear team new paragraph thanks", ["new_paragraph"]),
    ("our site is example dot io", ["url"]),
    ("send it to j patel at outlook dot com", ["email"]),
    ("the the report is ready", ["repeated_words"]),
    ("hello comma world", ["comma"]),
    ("I think I think we should ship it", ["restart"]),
    ("it goes on and on and on and on", ["restart", "phrase_repeat"]),   # each alone makes the same fix
    ("send it Monday scratch that send it Tuesday", ["scratch_that"]),
    ("I came semi colon I saw", ["semicolon"]),                    # not also credited to colon
    ("call me period then the the report", ["period", "repeated_words"]),
])
def test_fired_rules(text, fired):
    assert sorted(ne.fired_rules(text, ne.default_cfg())) == sorted(fired)


def test_default_cfg_is_the_committed_default_not_local():
    from dictum.config import DEFAULT_PATH
    want = yaml.safe_load(DEFAULT_PATH.read_text())["refine"]["normalize"]
    assert ne.default_cfg() == want


def _case(id, rule, fire, text, expected, position=""):
    return ne.Case(id=id, rule=rule, fire=fire, input=text, expected=expected, position=position)


def test_misfire_is_charged_to_the_rule_that_fired_not_the_target():
    cases = [_case("a", "colon", False, "hello semicolon world", "hello semicolon world")]
    (row,) = ne.score(cases, ne.default_cfg())
    assert row["passed"] is False and row["fired"] == ["semicolon"]
    rules = ne.summarize([row])["rules"]
    assert rules["semicolon"]["fp"] == 1 and rules["colon"]["fp"] == 0


def test_metrics_arithmetic():
    cases = [
        _case("f1", "comma", True, "x comma y", "x, y"),                       # tp
        _case("f2", "comma", True, "x comma y", "WRONG"),                      # fn
        _case("n1", "comma", False, "hello comma world", "hello comma world"), # fp (comma fired)
        _case("n2", "comma", False, "plain words", "plain words"),             # tn
    ]
    s = ne.summarize(ne.score(cases, ne.default_cfg()))
    r = s["rules"]["comma"]
    assert (r["tp"], r["fn"], r["fp"], r["n_fire"], r["n_nofire"]) == (1, 1, 1, 2, 2)
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)
    assert 0 <= r["precision_ci"][0] <= 0.5 <= r["precision_ci"][1] <= 1
    assert s["overall"]["n"] == 4 and s["overall"]["passed"] == 2
    assert s["overall"]["precision"] == pytest.approx(0.5)


def test_unattributable_failure_is_reported_not_guessed():
    # Fails, but no rule fired (expected is simply wrong): must be flagged, not charged to a rule.
    s = ne.summarize(ne.score([_case("u", "comma", False, "plain words", "other words")], ne.default_cfg()))
    assert s["overall"]["unattributed_failures"] == 1
    assert s["rules"]["comma"]["fp"] == 0


def test_no_division_by_zero_when_a_rule_never_fires():
    s = ne.summarize(ne.score([_case("f", "comma", True, "x comma y", "x, y")], ne.default_cfg()))
    assert s["rules"]["period"]["precision"] is None and s["rules"]["period"]["recall"] is None


# ---------------------------------------------------------------- saving and CLI

@pytest.fixture
def tiny_cases(tmp_path, monkeypatch):
    for split in ne.SPLITS:
        (tmp_path / f"{split}.yaml").write_text(yaml.safe_dump([
            {"id": f"{split}-a", "rule": "comma", "fire": True, "input": "a comma b", "expected": "a, b"},
            {"id": f"{split}-b", "rule": "colon", "fire": False, "input": "add a colon here",
             "expected": "add a colon here"},
        ]))
    monkeypatch.setattr(ne, "CASES_DIR", tmp_path)
    monkeypatch.setattr(ne, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


def test_cli_dev_exits_zero_with_failures_shows_them_and_saves(tiny_cases):
    from dictum.cli import app
    res = CliRunner().invoke(app, ["eval-normalize", "--split", "dev"])
    assert res.exit_code == 0, res.output
    assert "out:" in res.output                       # the misfire is listed on dev
    (saved,) = (tiny_cases / "results").glob("*-normalize-dev.json")
    assert '"rows"' in saved.read_text()


def test_cli_test_split_hides_failures_by_default(tiny_cases):
    from dictum.cli import app
    res = CliRunner().invoke(app, ["eval-normalize", "--split", "test"])
    assert res.exit_code == 0, res.output
    assert "failures hidden" in res.output and "out:" not in res.output


def test_cli_rejects_unknown_split(tiny_cases):
    from dictum.cli import app
    assert CliRunner().invoke(app, ["eval-normalize", "--split", "nope"]).exit_code != 0
