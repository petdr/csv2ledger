"""Stop a re-run from double-posting transactions.

The wrapper scripts download overlapping date ranges (`--skip-older-than 7`), so the same
row arrives in more than one CSV. Two independent checks catch that:

  * an exact fingerprint log of everything this tool has committed -- reliable, and the
    one that makes a straight re-import a no-op
  * a scan of the target journal for a transaction on the same date for the same amount --
    a heuristic, because the journal keeps the payee rather than the bank description, so
    it can only warn, never skip

Both count occurrences rather than storing a bare set: two identical coffees on the same
day are two real transactions, and the second must not be swallowed as a duplicate of the
first.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from .model import Transaction


class FingerprintLog:
    """Append-only record of the transactions this tool has already emitted.

    Plain text rather than JSON: the trailing fields are there so a human scanning the
    file can see what a fingerprint refers to, and only the first field is ever parsed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._counts: Counter[str] | None = None
        # What this run has committed, kept apart from what previous runs did: a row is a
        # duplicate only until it has been emitted as many times as it occurs.
        self._committed: Counter[str] = Counter()

    @property
    def counts(self) -> Counter[str]:
        if self._counts is None:
            self._counts = self._load()
        return self._counts

    def _load(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        if not self.path.is_file():
            return counts
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                token = line.split("\t", 1)[0].strip()
                if token and not token.startswith("#"):
                    counts[token] += 1
        return counts

    def seen(self, transaction: Transaction) -> bool:
        """True if a previous run imported this row and this run has not re-emitted it."""
        fingerprint = transaction.fingerprint()
        return self._committed[fingerprint] < self.counts[fingerprint]

    def mark(self, transaction: Transaction) -> None:
        """Note that this run has emitted the row, without persisting anything."""
        self._committed[transaction.fingerprint()] += 1

    def record(self, transaction: Transaction, payee: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        amount = transaction.credit or ("-" + transaction.debit if transaction.debit else "")
        line = "\t".join(
            [transaction.fingerprint(), transaction.ledger_date, amount, payee]
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.mark(transaction)


_DATE_LINE = re.compile(r"^(\d{4})[/-](\d{2})[/-](\d{2})")
_AMOUNT = re.compile(r"(-?[\d,]+\.\d{2})\s*$")


def journal_index(path: Path | None) -> Counter[tuple[str, str]]:
    """Count (date, absolute amount) pairs already present in a journal.

    Used only to flag a possible duplicate to the user. Amounts are compared as absolute
    values because which side of the entry carries the number depends on the template.
    """
    index: Counter[tuple[str, str]] = Counter()
    if path is None or not path.is_file():
        return index

    current: str | None = None
    # Both postings of a balanced entry carry the same figure; counting each once per
    # transaction keeps the count meaning "how many transactions", not "how many lines".
    within: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            if not line[:1].isspace():
                match = _DATE_LINE.match(line)
                current = f"{match.group(1)}/{match.group(2)}/{match.group(3)}" if match else None
                within = set()
                continue
            if current is None:
                continue
            amount = _AMOUNT.search(line.rstrip())
            if amount:
                value = amount.group(1).replace(",", "").lstrip("-")
                if value not in within:
                    within.add(value)
                    index[(current, value)] += 1
    return index


def journal_key(transaction: Transaction) -> tuple[str, str]:
    amount = transaction.credit or transaction.debit
    return (transaction.ledger_date, amount.replace(",", "").lstrip("-"))
