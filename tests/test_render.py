"""Template rendering, which has to match what the journal already contains."""

from __future__ import annotations

from datetime import date

from csv2ledger.config import DEFAULT_TEMPLATE, BankConfig
from csv2ledger.model import Transaction
from csv2ledger.render import journal_entry, write_entries


def _bank(**overrides) -> BankConfig:
    settings: dict = {"name": "TEST", "account": "Liabilities:Credit Card:Test"}
    settings.update(overrides)
    return BankConfig(**settings)


def _transaction(credit: str = "", debit: str = "4.50") -> Transaction:
    return Transaction(
        bank="TEST",
        ledger_date="2026/02/01",
        entry_date=date(2026, 2, 1),
        description="COFFEE SHOP",
        credit=credit,
        debit=debit,
        raw_csv="01/02/2026,COFFEE SHOP,4.50",
        row_index=1,
    )


class TestEntry:
    def test_header_line(self):
        entry = journal_entry(
            _transaction(), "Coffee Shop", "Expenses:Food", _bank(), template=DEFAULT_TEMPLATE
        )
        assert entry.splitlines()[1] == "2026/02/01 * Coffee Shop"

    def test_currency_is_blank_on_the_side_with_no_amount(self):
        """This is what produces the one-sided postings ledger balances for you."""
        entry = journal_entry(
            _transaction(), "Coffee Shop", "Expenses:Food", _bank(), template=DEFAULT_TEMPLATE
        )
        lines = entry.splitlines()
        assert lines[2].endswith("AUD 4.50")
        assert lines[3].rstrip().endswith("Liabilities:Credit Card:Test")

    def test_account_column_is_padded_to_sixty(self):
        entry = journal_entry(
            _transaction(), "Coffee Shop", "Expenses:Food", _bank(), template=DEFAULT_TEMPLATE
        )
        assert entry.splitlines()[2].startswith("    " + "Expenses:Food".ljust(60))

    def test_cleared_character_is_configurable(self):
        entry = journal_entry(
            _transaction(),
            "Coffee Shop",
            "Expenses:Food",
            _bank(cleared_character="!"),
            template=DEFAULT_TEMPLATE,
        )
        assert entry.splitlines()[1].startswith("2026/02/01 ! ")

    def test_credit_currency_can_differ(self):
        entry = journal_entry(
            _transaction(credit="9.00", debit=""),
            "Refund",
            "Expenses:Food",
            _bank(credit_currency="EUR"),
            template=DEFAULT_TEMPLATE,
        )
        assert "EUR 9.00" in entry

    def test_tags_are_indented_and_joined(self):
        entry = journal_entry(
            _transaction(),
            "Coffee Shop",
            "Expenses:Food",
            _bank(),
            template="{payee}\n    ; {tags}\n",
            tags=["one", "two"],
        )
        assert entry == "Coffee Shop\n    ; one\n    ; two\n"

    def test_unknown_placeholder_names_the_available_ones(self):
        try:
            journal_entry(
                _transaction(), "X", "Y", _bank(), template="{nonsense}"
            )
        except KeyError as exc:
            assert "nonsense" in str(exc) and "payee" in str(exc)
        else:
            raise AssertionError("expected a KeyError")


class TestWriteEntries:
    def test_entries_are_newline_separated(self, tmp_path):
        path = tmp_path / "out.ledger"
        with path.open("w", encoding="utf-8") as handle:
            write_entries(["a", "b"], handle)
        assert path.read_text(encoding="utf-8") == "a\nb\n"

    def test_no_entries_writes_nothing(self, tmp_path):
        path = tmp_path / "out.ledger"
        with path.open("w", encoding="utf-8") as handle:
            write_entries([], handle)
        assert path.read_text(encoding="utf-8") == ""
