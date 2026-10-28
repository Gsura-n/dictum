"""Correction-cue pre-pass and filler-aware chunking.

Parakeet writes "Tuesday. Um no wait uh Thursday." The fine-tuned model keeps the cue when punctuation sits in
front of it, so normalize removes the filler and the boundary punctuation before four unambiguous cues and
leaves the actual retraction to the model. Phrases that merely look like cues must come out unchanged.
"""
import pytest

from dictum import normalize as n


@pytest.mark.parametrize("raw,want", [
    ("Let's meet on Tuesday. Um no wait uh Thursday at three.",
     "Let's meet on Tuesday no wait uh Thursday at three."),
    ("Send it to the team lead. Oh actually no, send it to the whole team.",
     "Send it to the team lead actually no, send it to the whole team."),
    ("I'll have the report ready by Monday. Sorry, I mean Wednesday.",
     "I'll have the report ready by Monday sorry, I mean Wednesday."),
    ("We should use Postgres. Or rather, we should use SQLite for this.",
     "We should use Postgres or rather, we should use SQLite for this."),
    ("I'll send the invoice tomorrow, or rather, on Thursday after the review.",
     "I'll send the invoice tomorrow or rather, on Thursday after the review."),
    ("The budget is five thousand dollars. No wait, fifteen thousand dollars.",
     "The budget is five thousand dollars no wait, fifteen thousand dollars."),
    ("Let's meet on Tuesday, well, no wait, Thursday.", "Let's meet on Tuesday no wait, Thursday."),
    ("Meet at nine; actually no, ten.", "Meet at nine actually no, ten."),
    ("Meet at nine. Uh, um, actually, no, ten.", "Meet at nine actually, no, ten."),
])
def test_cue_after_boundary_is_joined_to_what_it_retracts(raw, want):
    assert n.correction_cues(raw) == want


@pytest.mark.parametrize("text", [
    "There was no wait at the counter.",
    "I can't make that meeting.",
    "That's right. No, I agree.",
    "No, I don't think so.",
    "Sorry, I'm late.",
    "I mean it.",
    "Thanks. Sorry, I mean it.",                          # an apology, not a retraction
    "Next. No wait times are posted.",                     # "wait" is a noun here
    "It works. Actually no longer needed.",
    "Pick red. Or rather than that, blue.",
    "Is it Tuesday? No wait, Thursday.",                   # a question mark is meaning, not a stray period
    "No wait, Thursday at three.",                         # nothing in front to join to
    "Send it to the team lead no wait the whole team.",    # already one clause: nothing to strip
    "Please call Dave. Scratch that, call Mike.",          # scratch that has its own rule
    "Tuesday. Make that Thursday.",                        # not a cue here
    "Her name is Sara. No, not Sara, Sarah with an H.",    # bare "no" is not a cue
])
def test_look_alikes_and_out_of_scope_cues_unchanged(text):
    assert n.correction_cues(text) == text


def test_apply_runs_the_pre_pass_and_the_flag_turns_it_off():
    raw = "Let's meet on Tuesday. Um no wait uh Thursday at three."
    assert n.join(n.apply(raw)) == "Let's meet on Tuesday no wait uh Thursday at three."
    assert n.join(n.apply(raw, {"correction_cues": False})) == raw


def test_apply_keeps_paragraph_breaks_apart():
    raw = "Meet on Tuesday new paragraph no wait Thursday. Um no wait Friday."
    segs = n.apply(raw)
    assert [b for _, b in segs] == ["\n\n", ""]
    assert segs[1][0] == "no wait Thursday no wait Friday."


def _long_text(correction: str) -> tuple[str, str]:
    """~70 words: a 54-word sentence, the antecedent, the correction, then a closing sentence. The correction does
    not fit in the first 60-word chunk, so only the glue rule keeps it with the antecedent."""
    lead = ("so the plan for the release is that we finish the migration first and then we run the full regression "
            "suite and after that we write up the notes for the team and send them around so everyone knows what "
            "changed and what to watch for before we ship it on Friday afternoon")
    antecedent = "Send it to the team lead."
    return f"{lead}. {antecedent} {correction} Then we can move on to the budget.", antecedent


@pytest.mark.parametrize("correction", [
    "Oh actually no, send it to the whole team.",
    "Um no wait, send it to the whole team.",
    "Uh, sorry, send it to the whole team.",
    "Well no, send it to the whole team.",
    "Um, oh, actually send it to the whole team.",
    "Actually no, send it to the whole team.",            # no filler: worked before, must still
])
def test_chunk_keeps_filler_prefixed_corrections_with_their_antecedent(correction):
    text, antecedent = _long_text(correction)
    assert 70 <= len(text.split()) <= 80
    parts = n.chunk(text, 60)
    assert len(parts) >= 2
    assert " ".join(parts) == text
    hold = [p for p in parts if antecedent in p]
    assert len(hold) == 1 and correction in hold[0]


def test_chunk_does_not_glue_a_filler_sentence_that_is_not_a_correction():
    text, antecedent = _long_text("Um so I will send the notes myself.")
    parts = n.chunk(text, 60)
    assert len(parts) >= 2
    assert not any(antecedent in p and "Um so I will" in p for p in parts)
