"""CSV decoding, including the icsv2ledger quirks the differential test depends on."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from csv2ledger.config import BankConfig, ColumnRef, ConfigError
from csv2ledger.csvsource import CsvError, read


def _bank(**overrides) -> BankConfig:
    settings: dict = {
        "name": "TEST",
        "account": "Assets:Bank:Test",
        "date": ColumnRef(index=1),
        "description": (ColumnRef(index=2),),
        "debit": ColumnRef(index=3),
    }
    settings.update(overrides)
    return BankConfig(**settings)


def _write(tmp_path, text: str):
    path = tmp_path / "bank.csv"
    path.write_text(text, encoding="utf-8")
    return path


class TestAmounts:
    def test_negative_index_inverts_the_sign(self, tmp_path):
        """`debit = -4` is how ONE, EVERYDAY, MAX and RABO are configured."""
        path = _write(tmp_path, "01/02/2026,COFFEE,-4.50\n")
        bank = _bank(debit=ColumnRef(index=3, invert=True))
        assert next(read(path, bank)).debit == "4.50"

    def test_negative_index_makes_a_positive_negative(self, tmp_path):
        path = _write(tmp_path, "01/02/2026,COFFEE,4.50\n")
        bank = _bank(debit=ColumnRef(index=3, invert=True))
        assert next(read(path, bank)).debit == "-4.50"

    def test_parenthesised_amount_is_negative(self, tmp_path):
        path = _write(tmp_path, '01/02/2026,COFFEE,(13.37)\n')
        assert next(read(path, _bank())).debit == "-13.37"

    def test_currency_symbols_and_separators_are_stripped(self, tmp_path):
        path = _write(tmp_path, '01/02/2026,COFFEE,"$1,234.56"\n')
        assert next(read(path, _bank())).debit == "1234.56"

    def test_zero_side_is_dropped_when_both_columns_are_present(self, tmp_path):
        """A one-sided posting is what lets ledger balance the entry itself."""
        path = _write(tmp_path, "01/02/2026,COFFEE,0.00,4.50\n")
        bank = _bank(debit=ColumnRef(index=3), credit=ColumnRef(index=4))
        transaction = next(read(path, bank))
        assert (transaction.debit, transaction.credit) == ("", "4.50")

    def test_absent_column_yields_empty(self, tmp_path):
        path = _write(tmp_path, "01/02/2026,COFFEE,4.50\n")
        bank = _bank(credit=ColumnRef())
        assert next(read(path, bank)).credit == ""

    def test_decimal_comma_is_converted(self, tmp_path):
        path = _write(tmp_path, "01/02/2026;COFFEE;4,50\n")
        bank = _bank(delimiter=";", csv_decimal_comma=True)
        assert next(read(path, bank)).debit == "4.50"


class TestDates:
    def test_reformatted_into_the_ledger_format(self, tmp_path):
        path = _write(tmp_path, "01/02/2026,COFFEE,4.50\n")
        transaction = next(read(path, _bank()))
        assert transaction.ledger_date == "2026/02/01"
        assert transaction.entry_date == date(2026, 2, 1)

    def test_alternative_csv_format(self, tmp_path):
        path = _write(tmp_path, "2026-02-01,COFFEE,4.50\n")
        bank = _bank(csv_date_format="%Y-%m-%d")
        assert next(read(path, bank)).ledger_date == "2026/02/01"

    def test_mismatched_format_is_an_error_naming_the_row(self, tmp_path):
        path = _write(tmp_path, "not-a-date,COFFEE,4.50\n")
        with pytest.raises(CsvError, match="row 1"):
            next(read(path, _bank()))

    def test_skip_older_than_drops_old_rows_only(self, tmp_path):
        path = _write(
            tmp_path, "01/01/2026,OLD,1.00\n20/02/2026,NEW,2.00\n"
        )
        rows = list(
            read(path, _bank(), skip_older_than=7, today=datetime(2026, 2, 22))
        )
        assert [r.description for r in rows] == ["NEW"]


class TestColumns:
    def test_several_description_columns_are_joined(self, tmp_path):
        path = _write(tmp_path, "01/02/2026,COFFEE,SHOP,4.50\n")
        bank = _bank(
            description=(ColumnRef(index=2), ColumnRef(index=3)),
            debit=ColumnRef(index=4),
        )
        assert next(read(path, bank)).description == "COFFEE SHOP"

    def test_named_columns_resolve_against_the_header(self, tmp_path):
        path = _write(tmp_path, "Date,Narrative,Amount\n01/02/2026,COFFEE,4.50\n")
        bank = _bank(
            date=ColumnRef(name="Date"),
            description=(ColumnRef(name="Narrative"),),
            debit=ColumnRef(name="Amount"),
            skip_lines=1,
        )
        transaction = next(read(path, bank))
        assert (transaction.description, transaction.debit) == ("COFFEE", "4.50")

    def test_named_columns_require_the_header_to_be_skipped(self, tmp_path):
        path = _write(tmp_path, "Date,Narrative,Amount\n01/02/2026,COFFEE,4.50\n")
        bank = _bank(date=ColumnRef(name="Date"), skip_lines=0)
        with pytest.raises(ConfigError, match="skip_lines"):
            next(read(path, bank))

    def test_unknown_column_name_is_reported(self, tmp_path):
        path = _write(tmp_path, "Date,Narrative,Amount\n01/02/2026,COFFEE,4.50\n")
        bank = _bank(debit=ColumnRef(name="Nope"), skip_lines=1)
        with pytest.raises(ConfigError, match="not found in CSV header"):
            next(read(path, bank))


class TestFingerprint:
    def test_ignores_trailing_columns_the_bank_may_change(self, tmp_path):
        """Re-downloads differ in running balance; the fingerprint must not."""
        first = next(read(_write(tmp_path, "01/02/2026,COFFEE,4.50,100.00\n"), _bank()))
        second = next(
            read(_write(tmp_path, "01/02/2026,COFFEE,4.50,250.00\n"), _bank())
        )
        assert first.fingerprint() == second.fingerprint()
        assert first.md5sum != second.md5sum

    def test_differs_on_amount(self, tmp_path):
        first = next(read(_write(tmp_path, "01/02/2026,COFFEE,4.50\n"), _bank()))
        second = next(read(_write(tmp_path, "01/02/2026,COFFEE,5.50\n"), _bank()))
        assert first.fingerprint() != second.fingerprint()

    def test_ignores_whitespace_runs_in_the_description(self, tmp_path):
        first = next(read(_write(tmp_path, "01/02/2026,COFFEE  SHOP,4.50\n"), _bank()))
        second = next(read(_write(tmp_path, "01/02/2026,COFFEE SHOP,4.50\n"), _bank()))
        assert first.fingerprint() == second.fingerprint()
