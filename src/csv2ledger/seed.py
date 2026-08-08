"""Build the initial training corpus from data that already exists.

Two sources, with different shapes:

  * ``mappings/*`` -- roughly 6,300 unique (description, payee, account) rows. This is
    the only place a raw bank description is stored next to its label, so it is the
    only source that can train the description-to-payee model.
  * ``ross-family.dat`` -- ~34,000 journal transactions. These have no raw description,
    but they do record which account a payee is normally posted to, which makes a
    strong prior for the second stage of prediction.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import mappings
from .model import Example
from .normalize import from_regex, normalize

# Regex-derived examples are weaker evidence: the pattern is a description shape, not an
# observed transaction, so it gets a fractional weight in the nearest-neighbour vote.
REGEX_WEIGHT = 0.4

# A placeholder the user files things under when undecided. Training on it teaches the
# model to propose "I don't know", which is worse than proposing nothing.
PLACEHOLDER_ACCOUNTS = frozenset({"Expenses:Unknown"})


@dataclass
class SeedCorpus:
    examples: list[Example] = field(default_factory=list)
    # payee -> Counter of accounts it has been posted against in the journal
    payee_accounts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    known_payees: set[str] = field(default_factory=set)
    known_accounts: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def best_account_for(self, payee: str, exclude: set[str] | None = None) -> str | None:
        """Most common journal account for a payee, ignoring the bank's own side."""
        counts = self.payee_accounts.get(payee)
        if not counts:
            return None
        exclude = exclude or set()
        for account, _ in counts.most_common():
            if account not in exclude:
                return account
        return None


def from_mappings(paths: list[Path], corpus: SeedCorpus | None = None) -> SeedCorpus:
    """Turn mapping files into training examples.

    Rules are deduplicated first. The per-bank files share huge blocks of identical rows
    (mappings.COMMBANK and mappings.ONE have 3,465 in common); without deduping, a rule
    would be weighted by how many banks copied it rather than by how often it occurs.
    """
    corpus = corpus or SeedCorpus()
    report = mappings.read_all(paths)
    unique = mappings.dedupe(report.rules)

    corpus.notes.append(
        f"mappings: {len(report.rules)} rows across {len(report.files_read)} files, "
        f"{len(unique)} unique"
    )
    if report.skipped_short:
        corpus.notes.append(f"mappings: skipped {report.skipped_short} malformed row(s)")
    if report.skipped_bad_regex:
        corpus.notes.append(
            f"mappings: skipped {len(report.skipped_bad_regex)} invalid regex(es)"
        )

    for rule in unique:
        corpus.known_payees.add(rule.payee)
        corpus.known_accounts.add(rule.account)
        if rule.account in PLACEHOLDER_ACCOUNTS:
            continue

        if rule.regex is not None:
            text = from_regex(rule.pattern)
            weight = REGEX_WEIGHT
            origin = "mapping-regex"
        else:
            text = normalize(rule.pattern)
            weight = 1.0
            origin = "mapping"

        if not text:
            continue
        corpus.examples.append(
            Example(
                description=text,
                payee=rule.payee,
                account=rule.account,
                weight=weight,
                origin=origin,
                tags=rule.tags,
            )
        )
    return corpus


_TRANSACTION_START = re.compile(r"^(\d{4}[/-]\d{2}[/-]\d{2})\s*(?:([*!])\s*)?(.*)$")
_POSTING = re.compile(r"^\s+([A-Z][^;].*?)(?:\s\s+.*)?$")


def from_journal(path: Path, corpus: SeedCorpus | None = None) -> SeedCorpus:
    """Learn the payee-to-account prior from an existing ledger journal.

    Only the account vocabulary and payee/account co-occurrence are extracted; the
    journal does not retain the original bank description, so it cannot train the
    description model.
    """
    corpus = corpus or SeedCorpus()
    if not path.is_file():
        corpus.notes.append(f"journal: {path} not found, skipping")
        return corpus

    transactions = 0
    payee: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith((";", "#")):
                continue

            start = _TRANSACTION_START.match(line)
            if start:
                payee = start.group(3).strip() or None
                if payee:
                    transactions += 1
                    corpus.known_payees.add(payee)
                continue

            if payee is None or not line[:1].isspace():
                continue
            posting = _POSTING.match(line.rstrip())
            if not posting:
                continue
            account = posting.group(1).strip()
            # Strip a trailing amount that had only one space before it.
            account = re.sub(r"\s+(AUD|EUR|USD|GBP|NZD)\s+-?[\d,.]+$", "", account).strip()
            if not account or account.startswith(("assert", "check")):
                continue
            corpus.known_accounts.add(account)
            corpus.payee_accounts[payee][account] += 1

    corpus.notes.append(
        f"journal: {transactions} transactions, {len(corpus.known_accounts)} accounts, "
        f"{len(corpus.payee_accounts)} payees with account history"
    )
    return corpus


def build(mapping_paths: list[Path], journal: Path | None) -> SeedCorpus:
    """Assemble the full seed corpus from every available source."""
    corpus = SeedCorpus()
    from_mappings(mapping_paths, corpus)
    if journal is not None:
        from_journal(journal, corpus)
    corpus.notes.append(
        f"corpus: {len(corpus.examples)} training examples, "
        f"{len(corpus.known_payees)} payees, {len(corpus.known_accounts)} accounts"
    )
    return corpus


def all_mapping_files(directory: Path) -> list[Path]:
    """Every mapping file in a directory, in a stable order."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("mappings.*") if p.is_file())
