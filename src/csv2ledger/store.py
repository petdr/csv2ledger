"""The learning store: a single append-only log of confirmed classifications.

One shared JSONL file replaces the per-bank mapping files. That is the fix for the
duplication in the old setup, where mappings.COMMBANK and mappings.ONE carried 3,465
identical rows because a merchant taught on one card was invisible to the other. Here a
confirmation is recorded once and every bank can use it.

JSONL rather than a database: it stays greppable, diffable and git-trackable, which is
how the mapping files were used, and it is append-only so a crash mid-import cannot
corrupt earlier entries.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from .model import Example
from .normalize import normalize

SCHEMA_VERSION = 1


@dataclass
class Confirmation:
    """One decision the user made about one transaction."""

    description: str
    normalized: str
    payee: str
    account: str
    bank: str
    direction: str
    at: str
    # True when the user changed the suggestion. These are the interesting rows: they
    # are where the model was wrong, and they are what a future error analysis needs.
    overridden: bool
    suggested_payee: str = ""
    suggested_account: str = ""
    confidence: float = 0.0
    tags: list[str] | None = None
    # How many times this decision was made. Always 1 as written during an import; only
    # ``compact`` ever raises it, by folding repeats of the same decision into one row.
    # Carrying the count is what lets compaction shrink the file without changing any
    # suggestion, since the suggester votes by weight.
    count: int = 1

    def to_json(self) -> str:
        payload = {
            "v": SCHEMA_VERSION,
            "description": self.description,
            "normalized": self.normalized,
            "payee": self.payee,
            "account": self.account,
            "bank": self.bank,
            "direction": self.direction,
            "at": self.at,
            "overridden": self.overridden,
        }
        if self.overridden:
            payload["suggested_payee"] = self.suggested_payee
            payload["suggested_account"] = self.suggested_account
            payload["confidence"] = round(self.confidence, 4)
        if self.tags:
            payload["tags"] = self.tags
        if self.count != 1:
            payload["count"] = self.count
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def to_example(self) -> Example:
        return Example(
            description=self.normalized,
            payee=self.payee,
            account=self.account,
            bank=self.bank,
            direction=self.direction,
            weight=float(self.count),
            origin="confirmed",
            tags=self.tags or [],
        )


class TrainingStore:
    """Append-only JSONL log of confirmations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> list[Confirmation]:
        """Read every confirmation, skipping rows that cannot be parsed.

        A single corrupt line -- a partial write from a killed process, say -- must not
        make the whole history unreadable.
        """
        if not self.path.is_file():
            return []
        confirmations: list[Confirmation] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    confirmations.append(
                        Confirmation(
                            description=payload["description"],
                            normalized=payload.get("normalized")
                            or normalize(payload["description"]),
                            payee=payload["payee"],
                            account=payload["account"],
                            bank=payload.get("bank", ""),
                            direction=payload.get("direction", ""),
                            at=payload.get("at", ""),
                            overridden=bool(payload.get("overridden", False)),
                            suggested_payee=payload.get("suggested_payee", ""),
                            suggested_account=payload.get("suggested_account", ""),
                            confidence=float(payload.get("confidence", 0.0)),
                            tags=payload.get("tags"),
                            count=int(payload.get("count", 1)),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return confirmations

    def append(self, confirmation: Confirmation) -> None:
        """Append one confirmation, flushed to disk before returning.

        Durability matters here: an interrupted import must not lose the classifications
        already made, or the user has to redo that work.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(confirmation.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def examples(self) -> list[Example]:
        return [c.to_example() for c in self.load()]

    def compact(self) -> int:
        """Collapse repeated confirmations of the same decision into one counted row.

        Suggestions must come out identical afterwards, which is the whole constraint on
        this operation. The suggester votes by weight, so the repeats cannot simply be
        dropped: a description confirmed as 'Woolies' fifty times and 'Woolworths' once
        would collapse to one row each and become a 50/50 tie, silently discarding a
        decade of evidence and reversing whatever the user had settled on. Summing the
        repeats into ``count``, which ``to_example`` passes through as the example weight,
        keeps the vote exactly as it was and only shrinks the file.

        Written via a temporary file and an atomic rename so an interrupted compaction
        cannot lose the log.
        """
        confirmations = self.load()
        if not confirmations:
            return 0
        latest: dict[tuple[str, str, str], Confirmation] = {}
        for confirmation in confirmations:
            key = (confirmation.normalized, confirmation.payee, confirmation.account)
            previous = latest.get(key)
            # The newest row wins on every field except the count, which accumulates.
            latest[key] = replace(
                confirmation,
                count=confirmation.count + (previous.count if previous else 0),
            )
        kept = list(latest.values())
        if len(kept) == len(confirmations):
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        )
        try:
            with handle:
                for confirmation in kept:
                    handle.write(confirmation.to_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        return len(confirmations) - len(kept)


def now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
