"""The normalize case files are well formed and balanced. These tests do NOT assert that normalize gets
the cases right: the known misfires fail today and the suite is measurement only."""
import re
from collections import Counter

import pytest

from dictum import normalize as n
from dictum import normalize_eval as ne

SPLITS = ("dev", "test")


@pytest.fixture(scope="module")
def cases():
    return {s: ne.load(s) for s in SPLITS}


def test_case_ids_unique_across_splits(cases):
    ids = [c.id for s in SPLITS for c in cases[s]]
    assert len(ids) == len(set(ids))


def test_every_rule_has_cases_in_both_splits(cases):
    for split in SPLITS:
        seen = Counter(c.rule for c in cases[split])
        for rule in ne.RULES:
            assert seen[rule] >= 2, f"{split}: rule {rule} has {seen[rule]} cases"


def test_each_split_is_roughly_half_should_fire(cases):
    for split in SPLITS:
        fire = sum(c.fire for c in cases[split])
        assert 0.4 <= fire / len(cases[split]) <= 0.65, f"{split}: {fire}/{len(cases[split])} should-fire"


def test_should_not_fire_cases_expect_their_input_unchanged(cases):
    for split in SPLITS:
        for c in cases[split]:
            if not c.fire:
                assert c.expected == c.input, c.id


def test_mid_clause_cases_for_comma_period_new_line(cases):
    """A clause-boundary-only fix must not look better than it is."""
    for split in SPLITS:
        for rule in ("comma", "period", "new_line"):
            mid = [c for c in cases[split] if c.rule == rule and c.fire and c.position == "mid"]
            assert len(mid) >= 2, f"{split}: only {len(mid)} mid-clause should-fire cases for {rule}"


def test_position_matches_the_input(cases):
    trigger_end = re.compile(r"(comma|period|(?:new|next) line)\s*$", re.IGNORECASE)
    for split in SPLITS:
        for c in cases[split]:
            if c.position in ("mid", "end") and c.rule in ("comma", "period", "new_line"):
                assert bool(trigger_end.search(c.input)) == (c.position == "end"), c.id


def test_trace_output_equals_normalize_apply(cases):
    """Drift guard: the scorer must measure exactly what the pipeline runs."""
    cfg = ne.default_cfg()
    for split in SPLITS:
        for c in cases[split]:
            assert ne.run(c.input, cfg) == n.join(n.apply(c.input, cfg)), c.id
