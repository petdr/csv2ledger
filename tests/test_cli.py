"""End-to-end runs of the import command against a throwaway config."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from csv2ledger.cli import _drift, main
from csv2ledger.model import Example
from csv2ledger.seed import SeedCorpus
from csv2ledger.store import TrainingStore
from csv2ledger.suggest import Suggester

CSV = """01/02/2026,ALDI STORES 1234 CARLTON VIC,15.00
01/02/2026,ALDI STORES 5678 FITZROY VIC,22.50
"""

MAPPINGS = "ALDI STORES 1234 CARLTON VIC,Aldi,Expenses:Food:Groceries\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "mappings").mkdir()
    (tmp_path / "mappings" / "mappings.TEST").write_text(MAPPINGS, encoding="utf-8")
    (tmp_path / "bank.csv").write_text(CSV, encoding="utf-8")
    (tmp_path / "banks.toml").write_text(
        f"""
[defaults]
root = "{tmp_path}"
mappings_dir = "mappings"
training_file = "training.jsonl"
fingerprint_file = "imported.log"

[banks.TEST]
account = "Liabilities:Credit Card:Test"
legacy_mapping_file = "mappings/mappings.TEST"
date = 1
description = 2
debit = 3
default_account = "Expenses:Unknown"
""",
        encoding="utf-8",
    )
    return tmp_path


def _run(workspace: Path, *extra: str, stdin: str = "") -> int:
    import sys

    original = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        return main(
            [
                "import",
                "-c",
                str(workspace / "banks.toml"),
                "-a",
                "TEST",
                str(workspace / "bank.csv"),
                "-o",
                str(workspace / "out.ledger"),
                *extra,
            ]
        )
    finally:
        sys.stdin = original


class TestImport:
    def test_accepting_every_suggestion_writes_both_entries(self, workspace):
        assert _run(workspace, "--auto", "0.95", stdin="\n" * 10) == 0
        output = (workspace / "out.ledger").read_text(encoding="utf-8")
        assert output.count("Liabilities:Credit Card:Test") == 2
        assert "Aldi" in output

    def test_the_second_variant_learns_from_the_first(self, workspace):
        """The store-number variant should be answered by the row just confirmed."""
        _run(workspace, stdin="\n" * 10)
        confirmations = TrainingStore(workspace / "training.jsonl").load()
        assert [c.payee for c in confirmations] == ["Aldi", "Aldi"]
        assert all(not c.overridden for c in confirmations)

    def test_an_override_is_recorded_as_one(self, workspace):
        _run(workspace, stdin="Aldi Supermarket\n\nAldi Supermarket\n\n")
        confirmations = TrainingStore(workspace / "training.jsonl").load()
        assert confirmations[0].payee == "Aldi Supermarket"
        assert confirmations[0].overridden
        assert confirmations[0].suggested_payee == "Aldi"

    def test_a_second_run_skips_what_was_already_imported(self, workspace):
        _run(workspace, stdin="\n" * 10)
        _run(workspace, stdin="\n" * 10)
        assert (workspace / "out.ledger").read_text(encoding="utf-8") == ""

    def test_dry_run_records_nothing(self, workspace):
        _run(workspace, "--dry-run", stdin="\n" * 10)
        assert not (workspace / "training.jsonl").exists()
        assert not (workspace / "imported.log").exists()

    def test_no_learn_still_writes_entries(self, workspace):
        _run(workspace, "--no-learn", stdin="\n" * 10)
        assert (workspace / "out.ledger").read_text(encoding="utf-8").count("\n") > 4
        assert not (workspace / "training.jsonl").exists()

    def test_skipping_a_transaction_leaves_it_out(self, workspace):
        _run(workspace, stdin=":s\n\n\n")
        output = (workspace / "out.ledger").read_text(encoding="utf-8")
        assert output.count("Liabilities:Credit Card:Test") == 1

    def test_quitting_keeps_what_was_already_classified(self, workspace):
        _run(workspace, stdin="\n\n:q\n")
        output = (workspace / "out.ledger").read_text(encoding="utf-8")
        assert output.count("Liabilities:Credit Card:Test") == 1

    def test_rules_only_never_prompts(self, workspace):
        """The icsv2ledger-compatible path used by the differential test."""
        assert _run(workspace, "--rules-only") == 0
        output = (workspace / "out.ledger").read_text(encoding="utf-8")
        assert "Aldi" in output
        assert "Expenses:Unknown" in output  # the unmatched second row

    def test_retrain_collapses_repeated_confirmations(self, workspace, capsys):
        for _ in range(3):
            _run(workspace, "--no-dedupe", stdin="\n" * 10)
        assert len(TrainingStore(workspace / "training.jsonl").load()) == 6

        main(["retrain", "-c", str(workspace / "banks.toml")])
        assert len(TrainingStore(workspace / "training.jsonl").load()) == 2
        assert "4 duplicate(s) removed" in capsys.readouterr().out

    def test_retrain_rebuilds_the_index(self, workspace, capsys):
        _run(workspace, stdin="\n" * 10)
        assert main(["retrain", "-c", str(workspace / "banks.toml")]) == 0
        assert "index:" in capsys.readouterr().out

    def test_no_compact_leaves_the_store_alone(self, workspace, capsys):
        for _ in range(2):
            _run(workspace, "--no-dedupe", stdin="\n" * 10)
        main(["retrain", "-c", str(workspace / "banks.toml"), "--no-compact"])
        assert len(TrainingStore(workspace / "training.jsonl").load()) == 4
        assert "left untouched" in capsys.readouterr().out

    def test_retrain_survives_an_empty_store(self, workspace, capsys):
        assert main(["retrain", "-c", str(workspace / "banks.toml")]) == 0
        assert "nothing recorded yet" in capsys.readouterr().out

    def test_retrain_rejects_an_unknown_bank(self, workspace):
        assert main(["retrain", "-c", str(workspace / "banks.toml"), "-a", "NOPE"]) == 2

    def test_an_unknown_bank_is_an_error(self, workspace):
        assert (
            main(
                [
                    "import",
                    "-c",
                    str(workspace / "banks.toml"),
                    "-a",
                    "NOPE",
                    str(workspace / "bank.csv"),
                ]
            )
            == 2
        )


def _suggester(seeded: list[str], confirmed: list[str]) -> Suggester:
    examples = [
        Example(description=d, payee="P", account="Expenses:A", origin="mapping")
        for d in seeded
    ] + [
        Example(description=d, payee="P", account="Expenses:A", origin="confirmed")
        for d in confirmed
    ]
    return Suggester(SeedCorpus(examples=examples))


class TestDrift:
    def test_new_boilerplate_is_reported(self):
        """A bank changing its export format lands one token on every row at once."""
        seeded = [f"MERCHANT {n} MELBOURNE VI" for n in range(200)]
        confirmed = [f"OSKO MERCHANT {n} MELBOURNE VI" for n in range(60)]
        assert [token for token, _, _ in _drift(_suggester(seeded, confirmed))] == ["OSKO"]

    def test_a_run_of_new_merchants_is_not_reported(self):
        """Merchants arrive one at a time, so no single token dominates."""
        seeded = [f"MERCHANT {n} MELBOURNE VI" for n in range(200)]
        confirmed = [f"NEWSHOP{n} MELBOURNE VI" for n in range(60)]
        assert _drift(_suggester(seeded, confirmed)) == []

    def test_too_few_confirmations_to_judge(self):
        seeded = [f"MERCHANT {n} MELBOURNE VI" for n in range(200)]
        confirmed = [f"OSKO MERCHANT {n}" for n in range(10)]
        assert _drift(_suggester(seeded, confirmed)) == []
