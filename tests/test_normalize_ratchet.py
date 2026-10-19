"""Ratchet on the normalize dev split: it can only get better.

Scores dev only (the test split is never run from pytest). Every dev case must pass except the ones named in
KNOWN_FAILING. If a case outside the set fails, normalize regressed. If a case in the set starts passing, remove
it from the set so the bar goes up. Adding a case to KNOWN_FAILING lowers the bar and needs a human decision.
"""
from dictum import normalize_eval as ne

KNOWN_FAILING = {"period-nofire-02"}     # "the Jurassic period lasted": needs a noun check that costs recall


def test_dev_split_does_not_regress():
    rows = ne.score(ne.load("dev"), ne.default_cfg())
    failing = {r["id"] for r in rows if not r["passed"]}
    passed = sum(r["passed"] for r in rows)
    assert not failing - KNOWN_FAILING, f"regressed ({passed}/{len(rows)} pass): {sorted(failing - KNOWN_FAILING)}"
    assert not KNOWN_FAILING - failing, f"now passing, remove from KNOWN_FAILING: {sorted(KNOWN_FAILING - failing)}"
