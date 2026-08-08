"""The learning store and the duplicate guard."""

from __future__ import annotations

from datetime import date

from csv2ledger.dedupe import FingerprintLog, journal_index, journal_key
from csv2ledger.model import Transaction
from csv2ledger.store import Confirmation, TrainingStore, now_stamp
from collections import Counter


def _confirmation(payee: str = "Aldi", **overrides) -> Confirmation:
    settings: dict = {
        "description": "ALDI STORES 1234 CARLTON",
        "normalized": "ALDI STORES # CARLTON",
        "payee": payee,
        "account": "Expenses:Food:Groceries",
        "bank": "COMMBANK",
        "direction": "debit",
        "at": now_stamp(),
        "overridden": False,
    }
    settings.update(overrides)
    return Confirmation(**settings)


def _transaction(amount: str = "4.50", description: str = "COFFEE") -> Transaction:
    return Transaction(
        bank="TEST",
        ledger_date="2026/02/01",
        entry_date=date(2026, 2, 1),
        description=description,
        credit="",
        debit=amount,
        raw_csv="",
        row_index=1,
    )


class TestTrainingStore:
    def test_round_trip(self, tmp_path):
        store = TrainingStore(tmp_path / "training.jsonl")
        store.append(_confirmation())
        store.append(_confirmation(payee="Coles"))
        assert [c.payee for c in store.load()] == ["Aldi", "Coles"]

    def test_a_corrupt_line_does_not_hide_the_rest(self, tmp_path):
        """A killed process can leave a partial write; the history must survive it."""
        path = tmp_path / "training.jsonl"
        store = TrainingStore(path)
        store.append(_confirmation())
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"description": "truncated"\n')
        store.append(_confirmation(payee="Coles"))
        assert [c.payee for c in store.load()] == ["Aldi", "Coles"]

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert TrainingStore(tmp_path / "absent.jsonl").load() == []

    def test_suggestion_fields_are_only_written_for_overrides(self, tmp_path):
        accepted = _confirmation().to_json()
        overridden = _confirmation(
            overridden=True, suggested_payee="Aldi Mobile", confidence=0.8
        ).to_json()
        assert "suggested_payee" not in accepted
        assert "Aldi Mobile" in overridden

    def test_examples_carry_the_normalised_description(self, tmp_path):
        store = TrainingStore(tmp_path / "training.jsonl")
        store.append(_confirmation())
        example = store.examples()[0]
        assert example.description == "ALDI STORES # CARLTON"
        assert example.origin == "confirmed"

    def test_compact_keeps_one_row_per_distinct_decision(self, tmp_path):
        store = TrainingStore(tmp_path / "training.jsonl")
        for _ in range(3):
            store.append(_confirmation())
        store.append(_confirmation(payee="Coles"))
        assert store.compact() == 2
        assert [c.payee for c in store.load()] == ["Aldi", "Coles"]

    def test_compact_is_a_no_op_when_nothing_repeats(self, tmp_path):
        store = TrainingStore(tmp_path / "training.jsonl")
        store.append(_confirmation())
        assert store.compact() == 0

    def test_compact_preserves_the_vote(self, tmp_path):
        """Repeats are evidence. Collapsing them must not change what wins, or by how much.

        Dropping the repeats outright would turn a settled 50-to-1 preference into a
        50/50 tie and reverse it, which is a decade of decisions silently discarded.
        """
        store = TrainingStore(tmp_path / "training.jsonl")
        for _ in range(50):
            store.append(_confirmation("Woolies"))
        store.append(_confirmation("Woolworths"))

        def vote() -> dict[str, float]:
            tally: dict[str, float] = {}
            for example in store.examples():
                tally[example.payee] = tally.get(example.payee, 0.0) + example.weight
            return tally

        before = vote()
        assert before == {"Woolies": 50.0, "Woolworths": 1.0}
        store.compact()
        assert vote() == before
        assert len(store.load()) == 2

    def test_compact_is_idempotent(self, tmp_path):
        store = TrainingStore(tmp_path / "training.jsonl")
        for _ in range(5):
            store.append(_confirmation())
        assert store.compact() == 4
        assert store.compact() == 0
        assert store.load()[0].count == 5

    def test_the_count_survives_a_round_trip(self, tmp_path):
        store = TrainingStore(tmp_path / "training.jsonl")
        store.append(_confirmation(count=7))
        assert store.load()[0].count == 7
        assert store.examples()[0].weight == 7.0

    def test_a_count_of_one_is_left_out_of_the_json(self):
        """The common case stays as readable as it was before counts existed."""
        assert '"count"' not in _confirmation().to_json()
        assert '"count": 7' in _confirmation(count=7).to_json()


class TestFingerprintLog:
    def test_a_recorded_row_is_seen_on_the_next_run(self, tmp_path):
        log = FingerprintLog(tmp_path / "imported.log")
        transaction = _transaction()
        assert not log.seen(transaction)
        log.record(transaction, "Coffee Shop")
        assert FingerprintLog(tmp_path / "imported.log").seen(transaction)

    def test_recording_does_not_make_the_same_run_see_a_duplicate(self, tmp_path):
        log = FingerprintLog(tmp_path / "imported.log")
        transaction = _transaction()
        log.record(transaction, "Coffee Shop")
        assert not log.seen(transaction)

    def test_a_genuine_repeat_within_a_file_is_not_swallowed(self, tmp_path):
        """Two identical coffees on one day are two transactions, not a duplicate."""
        path = tmp_path / "imported.log"
        FingerprintLog(path).record(_transaction(), "Coffee Shop")

        log = FingerprintLog(path)
        transaction = _transaction()
        assert log.seen(transaction)  # the first matches the earlier import
        log.mark(transaction)
        assert not log.seen(transaction)  # the second is a new transaction

    def test_different_amounts_are_different_rows(self, tmp_path):
        log = FingerprintLog(tmp_path / "imported.log")
        log.record(_transaction("4.50"), "Coffee Shop")
        assert not log.seen(_transaction("5.50"))


class TestJournalIndex:
    def test_counts_a_two_sided_entry_once(self, tmp_path):
        path = tmp_path / "journal.dat"
        path.write_text(
            "2026/02/01 * Coffee Shop\n"
            "    Expenses:Food                    AUD 4.50\n"
            "    Liabilities:Credit Card:Test    AUD -4.50\n",
            encoding="utf-8",
        )
        assert journal_index(path)[("2026/02/01", "4.50")] == 1

    def test_matches_a_transaction_by_date_and_amount(self, tmp_path):
        path = tmp_path / "journal.dat"
        path.write_text(
            "2026/02/01 * Coffee Shop\n    Expenses:Food    AUD 4.50\n", encoding="utf-8"
        )
        assert journal_index(path)[journal_key(_transaction())] == 1

    def test_absent_journal_is_empty(self, tmp_path):
        assert journal_index(tmp_path / "nope.dat") == Counter()
