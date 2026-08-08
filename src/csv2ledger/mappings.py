"""Read icsv2ledger mapping files.

A mapping file is CSV: ``pattern,payee,account[,tags...]``. A pattern wrapped in
slashes is a regular expression matched with ``re.match`` (anchored at the start);
anything else must equal the description exactly.

These files are read-only to csv2ledger. They serve two purposes: they remain the
authoritative rule layer, and their ~6.3k unique rows are the training seed that makes
suggestions useful on day one.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rule:
    """One mapping-file row, ready to match against a description."""

    pattern: str
    payee: str
    account: str
    tags: list[str]
    regex: re.Pattern[str] | None
    source: str

    def matches(self, description: str) -> bool:
        if self.regex is not None:
            return self.regex.match(description) is not None
        return description == self.pattern


@dataclass
class LoadReport:
    """What happened while reading mapping files, so problems are visible not silent."""

    rules: list[Rule]
    skipped_short: int = 0
    skipped_bad_regex: list[str] = None  # type: ignore[assignment]
    files_read: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skipped_bad_regex is None:
            self.skipped_bad_regex = []
        if self.files_read is None:
            self.files_read = []


def read_file(path: str | Path, report: LoadReport | None = None) -> list[Rule]:
    """Parse one mapping file into Rules, skipping rows that cannot be used.

    Malformed rows are counted rather than fatal. icsv2ledger exits on a bad regex, but
    these files have accumulated damage over a decade (a payee field in mappings.SAV has
    swallowed the following rule), and refusing to start would make the good 6,000 rows
    unreachable because of a handful of bad ones.
    """
    source = Path(path)
    rules: list[Rule] = []
    if not source.is_file():
        return rules

    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                if report is not None and any(cell.strip() for cell in row):
                    report.skipped_short += 1
                continue
            pattern = row[0].strip()
            payee = row[1].strip()
            account = row[2].strip()
            tags = [tag.strip() for tag in row[3:] if tag.strip()]
            if not payee or not account:
                if report is not None:
                    report.skipped_short += 1
                continue

            compiled: re.Pattern[str] | None = None
            if pattern.startswith("/") and pattern.endswith("/") and len(pattern) > 1:
                try:
                    compiled = re.compile(pattern[1:-1])
                except re.error:
                    if report is not None:
                        report.skipped_bad_regex.append(f"{source.name}: {pattern}")
                    continue

            rules.append(
                Rule(
                    pattern=pattern,
                    payee=payee,
                    account=account,
                    tags=tags,
                    regex=compiled,
                    source=source.name,
                )
            )

    if report is not None:
        report.files_read.append(str(source))
    return rules


def read_all(paths: list[Path]) -> LoadReport:
    """Read several mapping files into one rule set.

    Order is preserved across files because icsv2ledger let a later rule override an
    earlier one, and some of these files rely on that.
    """
    report = LoadReport(rules=[])
    for path in paths:
        report.rules.extend(read_file(path, report))
    return report


def match(rules: list[Rule], description: str) -> Rule | None:
    """Return the winning rule for a description, or None.

    Last match wins, matching icsv2ledger, which scans the whole list without breaking
    so that later entries override earlier ones.
    """
    winner: Rule | None = None
    for rule in rules:
        if rule.matches(description):
            winner = rule
    return winner


def dedupe(rules: list[Rule]) -> list[Rule]:
    """Collapse rules identical in pattern, payee and account.

    The per-bank files carry heavy duplication (mappings.COMMBANK and mappings.ONE share
    3,465 rows). Deduping matters for training: without it the shared rows are weighted
    by how many banks happened to copy them rather than by how often they occur.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[Rule] = []
    for rule in rules:
        key = (rule.pattern, rule.payee, rule.account)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique
