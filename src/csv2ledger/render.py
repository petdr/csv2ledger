"""Render Transactions as ledger-cli journal entries via a format template.

The template placeholders match icsv2ledger's, so an existing TEMPLATE file works
unchanged. The important subtlety is that ``debit_currency`` is blank when there is no
debit amount (and likewise for credit) -- that is what produces the one-sided postings
in the existing journal, where ledger infers the balancing amount.
"""

from __future__ import annotations

from .config import BankConfig
from .model import Transaction


def journal_entry(
    transaction: Transaction,
    payee: str,
    account: str,
    bank: BankConfig,
    *,
    template: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Format one ledger entry."""
    tags = tags or []
    text = template if template is not None else bank.template()

    fields = {
        "date": transaction.ledger_date,
        "effective_date": "",
        "cleared_character": bank.cleared_character,
        "payee": payee,
        "transaction_index": transaction.row_index,
        "uuid": "",
        "debit_account": account,
        "debit_currency": bank.currency if transaction.debit else "",
        "debit": transaction.debit,
        "credit_account": bank.account,
        "credit_currency": bank.effective_credit_currency() if transaction.credit else "",
        "credit": transaction.credit,
        "tags": "\n    ; ".join(tags),
        "md5sum": transaction.md5sum,
        "csv": transaction.raw_csv,
    }
    try:
        return text.format(**fields)
    except KeyError as exc:
        raise KeyError(
            f"template references unknown placeholder {exc.args[0]!r}; "
            f"available: {', '.join(sorted(fields))}"
        ) from exc


def write_entries(entries: list[str], out) -> None:
    """Write entries the way icsv2ledger did: newline-separated, trailing newline."""
    if entries:
        print(*entries, sep="\n", file=out)
