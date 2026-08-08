"""Core data types shared across the package."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Transaction:
    """One row of a bank CSV, decoded but not yet classified.

    ``credit`` and ``debit`` are kept as strings rather than Decimals because they are
    written straight into the ledger output, and the exact text (trailing zeros, sign
    placement) is part of matching the old tool's output byte-for-byte.
    """

    bank: str
    ledger_date: str
    entry_date: date
    description: str
    credit: str
    debit: str
    raw_csv: str
    row_index: int

    @property
    def signed_amount(self) -> float:
        """Amount as a float, positive for money in, negative for money out.

        Used only for display and as a model feature, never for ledger output.
        """
        text = self.credit or ("-" + self.debit if self.debit else "")
        try:
            return float(text)
        except ValueError:
            return 0.0

    @property
    def direction(self) -> str:
        """A coarse categorical feature: which side of the account this row moves."""
        return "credit" if self.credit else "debit"

    @property
    def md5sum(self) -> str:
        return hashlib.md5(self.raw_csv.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        """Stable identity for duplicate detection across overlapping downloads.

        Deliberately built from date, amount and description rather than the raw CSV
        line: banks re-issue the same transaction with different trailing columns
        (running balance, receipt id), and those must not defeat the check.
        """
        amount = self.credit or ("-" + self.debit if self.debit else "")
        parts = f"{self.ledger_date}|{amount}|{' '.join(self.description.split())}"
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> str:
        amount = self.credit if self.credit else "-" + self.debit
        return f"{self.ledger_date} {self.description:<40} {amount}"


@dataclass
class Suggestion:
    """A ranked prediction, with enough provenance to explain itself."""

    payee: str
    account: str
    confidence: float
    source: str
    evidence: str = ""


@dataclass
class Example:
    """One labelled training example: a description mapped to a payee and account."""

    description: str
    payee: str
    account: str
    bank: str = ""
    direction: str = ""
    # Regex-derived examples are weaker evidence than a real confirmed transaction.
    weight: float = 1.0
    origin: str = "seed"
    tags: list[str] = field(default_factory=list)
