"""Decode a bank CSV into Transactions.

Field extraction reproduces icsv2ledger's ``get_field_at_index`` exactly, including the
quirks, because the differential test requires byte-identical output:

  * everything outside ``[-0-9.]`` is stripped from an amount
  * ``(13.37)`` is read as ``-13.37``
  * a negative column index inverts the sign of whatever is left
  * when both credit and debit are present and one of them is zero, the zero is dropped
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from .config import BankConfig, ColumnRef, ConfigError
from .model import Transaction


class CsvError(Exception):
    """Raised when a CSV row cannot be decoded under the bank's configuration."""


def _extract_amount(
    fields: list[str],
    ref: ColumnRef,
    header: list[str] | None,
    csv_decimal_comma: bool,
    ledger_decimal_comma: bool,
) -> str:
    position = ref.resolve(header)
    if position < 0 or position >= len(fields):
        return ""

    separator = "," if csv_decimal_comma else "."
    raw = fields[position]

    # Accounting-style negatives: ($13.37) means -13.37.
    if raw.startswith("(") and raw.endswith(")"):
        raw = "-" + raw[1:-1]

    value = re.sub("[^-0-9" + separator + "]", "", raw)

    if ref.invert and value:
        value = value[1:] if value.startswith("-") else "-" + value

    if csv_decimal_comma and not ledger_decimal_comma:
        value = value.replace(",", ".")
    elif not csv_decimal_comma and ledger_decimal_comma:
        value = value.replace(".", ",")

    return value


def _extract_description(
    fields: list[str], refs: tuple[ColumnRef, ...], header: list[str] | None
) -> str:
    parts = []
    for ref in refs:
        position = ref.resolve(header)
        if 0 <= position < len(fields):
            parts.append(fields[position].strip())
    return " ".join(parts).strip()


def _decode_row(
    fields: list[str],
    raw_line: str,
    row_index: int,
    bank: BankConfig,
    header: list[str] | None,
) -> Transaction:
    date_position = bank.date.resolve(header)
    if date_position < 0 or date_position >= len(fields):
        raise CsvError(f"row {row_index}: date column is out of range for {fields}")

    date_text = fields[date_position].strip()
    try:
        entry_date = datetime.strptime(date_text, bank.csv_date_format)
    except ValueError as exc:
        raise CsvError(
            f"row {row_index}: date {date_text!r} does not match "
            f"csv_date_format {bank.csv_date_format!r}"
        ) from exc

    ledger_date = (
        entry_date.strftime(bank.ledger_date_format) if bank.ledger_date_format else date_text
    )

    credit = _extract_amount(
        fields, bank.credit, header, bank.csv_decimal_comma, bank.ledger_decimal_comma
    )
    debit = _extract_amount(
        fields, bank.debit, header, bank.csv_decimal_comma, bank.ledger_decimal_comma
    )

    # A bank that fills both columns writes 0.00 in the unused one; drop it so the
    # rendered entry has a single amount and lets ledger balance the other side.
    if credit and debit:
        if _is_zero(credit):
            credit = ""
        elif _is_zero(debit):
            debit = ""

    return Transaction(
        bank=bank.name,
        ledger_date=ledger_date,
        entry_date=entry_date.date(),
        description=_extract_description(fields, bank.description, header),
        credit=credit,
        debit=debit,
        raw_csv=raw_line.strip(),
        row_index=row_index,
    )


def _is_zero(value: str) -> bool:
    try:
        return float(value.replace(",", ".")) == 0.0
    except ValueError:
        return False


def read(
    path: str | Path,
    bank: BankConfig,
    *,
    skip_older_than: int | None = None,
    today: datetime | None = None,
) -> Iterator[Transaction]:
    """Yield Transactions from a bank CSV.

    ``skip_older_than`` drops rows more than that many days old, matching the
    ``--skip-older-than`` flag the existing wrapper scripts pass.
    """
    source = Path(path).expanduser()
    text = source.read_text(encoding=bank.encoding)
    lines = text.splitlines(keepends=True)

    header = _header_for(lines, bank)
    body = lines[bank.skip_lines :]
    reader = csv.reader(body, delimiter=bank.delimiter)
    now = today or datetime.now()

    for offset, fields in enumerate(reader):
        if not fields:
            continue
        raw_line = body[offset] if offset < len(body) else ""
        transaction = _decode_row(fields, raw_line, offset + 1, bank, header)
        if skip_older_than is not None and skip_older_than >= 0:
            days_old = (now - datetime.combine(transaction.entry_date, datetime.min.time())).days
            if days_old > skip_older_than:
                continue
        yield transaction


def _header_for(lines: list[str], bank: BankConfig) -> list[str] | None:
    """Return the header row if the config needs one to resolve a named column.

    Only read when required, so a headerless CSV with purely numeric column references
    never has to justify its first line.
    """
    refs = (bank.date, bank.credit, bank.debit, *bank.description)
    if not any(ref.name for ref in refs):
        return None
    if bank.skip_lines < 1:
        raise ConfigError(
            f"bank {bank.name!r}: columns are referenced by name, so skip_lines must be at "
            f"least 1 so the header row is consumed"
        )
    header_line = lines[bank.skip_lines - 1]
    return next(csv.reader([header_line], delimiter=bank.delimiter), None)
