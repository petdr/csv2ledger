"""The completer's case handling.

A real corpus is not internally case-consistent -- the same "everyday" prefix covers
both machine-generated transfer payees (all caps, "EVERYDAY TO MAX") and merchant names
typed by hand ("Everyday Hero Events"). Case-folding everything, as a naive completer
does, mixes those two groups together and hands readline a common prefix of a single
letter, which silently discards whatever the user had typed.
"""

from __future__ import annotations

from csv2ledger.prompt import _Completer

OPTIONS = [
    "EVERYDAY TO MAX",
    "EVERYDAY TO PETER",
    "EVERYDAY TO EVERYDAY",
    "Everyday Hero Events",
    "Everyjewels",
    "Aldi",
]


def _matches(text: str) -> list[str]:
    completer = _Completer()
    completer.options = OPTIONS
    results = []
    state = 0
    while (match := completer.complete(text, state)) is not None:
        results.append(match)
        state += 1
    return results


def test_typed_capitals_are_matched_case_sensitively():
    # A capital in the query means "this is the group I mean" -- it must not pull in
    # the differently-cased group and dilute the common prefix.
    assert _matches("EVER") == ["EVERYDAY TO EVERYDAY", "EVERYDAY TO MAX", "EVERYDAY TO PETER"]


def test_typed_lowercase_matches_case_insensitively():
    # No capitals means the user is not distinguishing by case, so fold as before.
    assert _matches("ever") == [
        "EVERYDAY TO EVERYDAY",
        "EVERYDAY TO MAX",
        "EVERYDAY TO PETER",
        "Everyday Hero Events",
        "Everyjewels",
    ]


def test_typed_mixed_case_narrows_to_the_matching_group():
    assert _matches("Ever") == ["Everyday Hero Events", "Everyjewels"]


def test_no_match_falls_back_to_substring_in_the_typed_case():
    assert _matches("jewel") == ["Everyjewels"]
    # "Jewel" is not a case-sensitive substring of "Everyjewels" ("...jewels", lowercase
    # j) -- a capital in the query is a deliberate signal, so it should not be loosened
    # back to a case-insensitive match by the substring fallback either.
    assert _matches("Jewel") == []
