"""Reduce a bank description to a stable key.

This is the highest-leverage part of the system. Bank descriptions carry per-transaction
noise -- receipt numbers, card numbers, timestamps, store numbers, BSB/account pairs --
that makes every visit to the same merchant look like a new merchant. That is precisely
why the existing mapping files grew to ~6,300 rules: each variant needed its own line.

Digits are replaced with '#' rather than deleted, so "7-ELEVEN 1148" and "7-ELEVEN 3092"
collapse together while still recording that a number was present. The risk in the other
direction is real -- strip too much and distinct merchants merge -- so each rule below is
targeted at a specific observed pattern rather than being a blanket digit purge, and the
rules are pinned by tests against real descriptions.
"""

from __future__ import annotations

import re

# Ordered: earlier rules consume structure that later, broader rules would otherwise
# mangle. Card numbers before digit runs, dates before bare numbers, and so on.
_RULES: list[tuple[re.Pattern[str], str]] = [
    # ING embeds literal HTML in its CSV export.
    (re.compile(r"<br\s*/?>", re.IGNORECASE), " "),
    # Masked card numbers: 462263xxxxxx2739
    (re.compile(r"\b\d{6}x{4,6}\d{4}\b", re.IGNORECASE), " CARD "),
    # "Receipt 132547" -- and "Receipt 132547Data Processors", where the bank omits the
    # separator, so the digit run must not swallow the following word.
    (re.compile(r"\bReceipt\s*#?\s*\d+", re.IGNORECASE), " RECEIPT "),
    (re.compile(r"\bReceipt\s+number:?\s*\w+", re.IGNORECASE), " RECEIPT "),
    # "Date 19 Mar 2025", "Date 25/10/2025"
    (re.compile(r"\bDate\s+\d{1,2}[\s/-]\w{3,9}[\s/-]\d{2,4}", re.IGNORECASE), " "),
    (re.compile(r"\bTime\s+\d{1,2}:\d{2}\s*[AP]?M?", re.IGNORECASE), " "),
    # Standalone dates and times left over after the labelled forms above.
    (re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"), " "),
    (re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\s*[AP]?M?\b"), " "),
    (re.compile(r"\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b",
                re.IGNORECASE), " "),
    # Reference codes that are unique per transaction.
    (re.compile(r"\bRef:?\s*\w+", re.IGNORECASE), " "),
    (re.compile(r"\bCrn\s*\d+", re.IGNORECASE), " "),
    (re.compile(r"\bCard\s+\d+\b", re.IGNORECASE), " CARD "),
    # Trailing foreign-currency conversion: "FRANPRIX 4145 PARIS FR21.10 EUR".
    # No leading \b -- the country code is glued to the amount ("FR21.10"), so there is
    # no word boundary before the digits.
    (re.compile(r"\d+[.,]\d{2}\s*(AUD|EUR|USD|GBP|NZD|JPY|SGD|THB|CAD|CHF)\b",
                re.IGNORECASE), " "),
    # Cash-out and amount annotations.
    (re.compile(r"\bCash\s+(amount|out):?\s*\$?\s*[\d,]+\.?\d*", re.IGNORECASE), " CASHOUT "),
    (re.compile(r"\$\s*[\d,]+\.\d{2}"), " "),
    # BSB + account pairs, e.g. "To 063113 11267704", and any other long digit run.
    (re.compile(r"\b\d{6,}\b"), " # "),
    # Store, terminal and branch numbers: the short digit runs that remain.
    (re.compile(r"\b\d{2,5}\b"), " # "),
    # Single stray digits carry no signal once the above have run.
    (re.compile(r"\b\d\b"), " # "),
]

_COLLAPSE_HASHES = re.compile(r"(?:#\s*){2,}")
_PUNCTUATION = re.compile(r"[^\w#&*./'-]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(description: str) -> str:
    """Return the canonical form of a bank description.

    Idempotent: normalizing an already-normalized string returns it unchanged.
    """
    text = description.strip()
    if not text:
        return ""

    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)

    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _COLLAPSE_HASHES.sub("# ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text.upper()


# Regex metacharacters that carry no meaning once a pattern is treated as plain text.
_REGEX_NOISE = re.compile(r"[\\^$*+?(){}\[\]|]|\.\*|\.\+")


def from_regex(pattern: str) -> str:
    """Turn a mapping-file regex into a weak training string.

    A regex is a pattern, not an observed description, so it cannot be fed to the
    vectoriser directly. Stripping the metacharacters leaves the literal fragments --
    "/.*KATHMANDU PTY LIMITE/" becomes "KATHMANDU PTY LIMITE" -- which is real signal,
    just weaker than an actual transaction. Callers weight these examples down.
    """
    text = pattern
    if text.startswith("/") and text.endswith("/") and len(text) > 1:
        text = text[1:-1]
    text = _REGEX_NOISE.sub(" ", text)
    # Character classes reduce to nothing useful; drop what is left of them.
    text = re.sub(r"[A-Za-z]-[A-Za-z]", " ", text)
    return normalize(text)
