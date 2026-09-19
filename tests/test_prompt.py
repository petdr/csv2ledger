"""The prompt's readline integration: case-sensitive completion, and the libedit fallback.

A real corpus is not internally case-consistent -- the same "everyday" prefix covers
both machine-generated transfer payees (all caps, "EVERYDAY TO MAX") and merchant names
typed by hand ("Everyday Hero Events"). Case-folding everything, as a naive completer
does, mixes those two groups together and hands readline a common prefix of a single
letter, which silently discards whatever the user had typed.
"""

from __future__ import annotations

import builtins
import sys

from csv2ledger.prompt import Prompter, _Completer
from csv2ledger.suggest import Ranked

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


# -- the libedit pre-fill fallback -------------------------------------------------
#
# readline.set_startup_hook is how the suggestion is meant to land in the line buffer
# for a bare Enter to accept. libedit's readline compatibility shim accepts that call
# without error but never actually invokes the hook, so insert_text silently does
# nothing -- the suggestion would be invisible rather than pre-filled. _read detects
# this by backend name and prints the default in brackets instead. A fake readline
# module with a controllable ``backend`` lets both paths be tested regardless of what
# the real interpreter running the tests happens to link against.


class _FakeReadline:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def set_completer(self, fn) -> None:
        pass

    def set_completer_delims(self, delims: str) -> None:
        pass

    def set_startup_hook(self, hook=None) -> None:
        pass

    def insert_text(self, text: str) -> None:
        pass


def _prompter(monkeypatch, backend: str, typed: str) -> tuple[Prompter, list[str]]:
    import csv2ledger.prompt as prompt_module

    monkeypatch.setattr(prompt_module, "readline", _FakeReadline(backend))
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return typed

    monkeypatch.setattr(builtins, "input", fake_input)

    prompter = Prompter.__new__(Prompter)
    prompter._completer = _Completer()
    prompter._eof = False
    prompter.interactive = True
    prompter._stdout_is_tty = True
    prompter.out = sys.stderr
    return prompter, prompts


def test_gnu_readline_prefills_silently(monkeypatch):
    # The real mechanism works under GNU readline, so nothing needs to be printed --
    # the buffer already shows it.
    prompter, prompts = _prompter(monkeypatch, backend="readline", typed="")
    prompter._read("Payee", "Ben Wind indicator", {"Ben Wind indicator"})
    assert prompts == ["Payee   > "]


def test_editline_shows_the_default_in_brackets(monkeypatch):
    prompter, prompts = _prompter(monkeypatch, backend="editline", typed="")
    prompter._read("Payee", "Ben Wind indicator", {"Ben Wind indicator"})
    assert prompts == ["Payee   [Ben Wind indicator] > "]


def test_editline_with_no_default_omits_the_empty_brackets(monkeypatch):
    prompter, prompts = _prompter(monkeypatch, backend="editline", typed="")
    prompter._read("Payee", "", {"Ben Wind indicator"})
    assert prompts == ["Payee   > "]


def test_bare_enter_still_accepts_the_default_under_editline(monkeypatch):
    prompter, _ = _prompter(monkeypatch, backend="editline", typed="")
    ranked = Ranked([("Ben Wind indicator", 1.0, "exact")])
    result = prompter._field("Payee", "Ben Wind indicator", ranked, {"Ben Wind indicator"})
    assert result == "Ben Wind indicator"
