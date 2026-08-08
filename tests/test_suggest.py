"""The four suggestion layers, and the incremental learning that sits on top of them."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date

from csv2ledger.mappings import Rule
from csv2ledger.model import Example, Transaction
from csv2ledger.normalize import normalize
from csv2ledger.seed import SeedCorpus
from csv2ledger.suggest import Suggester

OWN = "Liabilities:Credit Card:Test"


def _example(description: str, payee: str, account: str, **overrides) -> Example:
    return Example(
        description=normalize(description), payee=payee, account=account, **overrides
    )


def _transaction(description: str) -> Transaction:
    return Transaction(
        bank="TEST",
        ledger_date="2026/02/01",
        entry_date=date(2026, 2, 1),
        description=description,
        credit="",
        debit="4.50",
        raw_csv="",
        row_index=1,
    )


def _corpus(examples: list[Example], **overrides) -> SeedCorpus:
    corpus = SeedCorpus(
        examples=examples,
        known_payees={e.payee for e in examples},
        known_accounts={e.account for e in examples},
    )
    for key, value in overrides.items():
        setattr(corpus, key, value)
    return corpus


class TestExactLayer:
    def test_an_identical_description_is_answered_with_certainty(self):
        corpus = _corpus([_example("WW METRO 8281 MELBOURNE VI", "Woolworths", "Expenses:Food")])
        suggester = Suggester(corpus)
        ranked = suggester.rank_payees(_transaction("WW METRO 8281 MELBOURNE VI"))
        assert ranked.top == ("Woolworths", 1.0, "exact")

    def test_store_number_and_suburb_do_not_break_the_match(self):
        """Normalisation is what turns thousands of variants into one key."""
        corpus = _corpus([_example("WW METRO 8281 MELBOURNE VI", "Woolworths", "Expenses:Food")])
        suggester = Suggester(corpus)
        ranked = suggester.rank_payees(_transaction("WW METRO 1234 MELBOURNE VI"))
        assert ranked.top is not None and ranked.top[0] == "Woolworths"

    def test_conflicting_history_reports_as_uncertain(self):
        """The same description has genuinely been filed both ways over the years."""
        description = "WW METRO 8281 MELBOURNE VI"
        corpus = _corpus(
            [
                _example(description, "Woolworths", "Expenses:Food"),
                _example(description, "Woolworths", "Expenses:Food"),
                _example(description, "Woolies", "Expenses:Food"),
            ]
        )
        ranked = Suggester(corpus).rank_payees(_transaction(description))
        assert ranked.top is not None
        assert ranked.top[0] == "Woolworths"
        assert ranked.top[1] < 0.7  # two of three votes, not a certainty


class TestRuleLayer:
    def test_a_regex_rule_answers_an_unseen_description(self):
        rules = [
            Rule(
                pattern="^UBER",
                payee="Uber",
                account="Expenses:Transport",
                tags=[],
                regex=re.compile("^UBER"),
                source="test",
            )
        ]
        suggester = Suggester(_corpus([]), rules=rules)
        ranked = suggester.rank_payees(_transaction("UBER TRIP HELP.UBER.COM"))
        assert ranked.top == ("Uber", 0.99, "rule")

    def test_exact_history_outranks_a_rule(self):
        description = "UBER EATS SYDNEY"
        rules = [
            Rule(
                pattern="^UBER",
                payee="Uber",
                account="Expenses:Transport",
                tags=[],
                regex=re.compile("^UBER"),
                source="test",
            )
        ]
        corpus = _corpus([_example(description, "Uber Eats", "Expenses:Food:Dining Out")])
        ranked = Suggester(corpus, rules=rules).rank_payees(_transaction(description))
        assert ranked.top is not None and ranked.top[0] == "Uber Eats"


class TestLiteralLayer:
    def test_a_known_payee_at_the_front_is_read_off_the_description(self):
        """The case nearest-neighbour is weakest at: a merchant seen only in the journal."""
        corpus = _corpus(
            [_example("SOMETHING ELSE ENTIRELY", "Other", "Expenses:Other")],
            known_payees={"Other", "City Beach"},
            payee_accounts={"City Beach": Counter({"Expenses:Clothing": 4})},
        )
        suggester = Suggester(corpus)
        ranked = suggester.rank_payees(_transaction("City Beach Highpoint Maribyrnong AUS"))
        assert ranked.top is not None and ranked.top[0] == "City Beach"

    def test_the_longest_matching_name_wins(self):
        corpus = _corpus([], known_payees={"Supa", "Supa IGA"})
        suggester = Suggester(corpus)
        ranked = suggester.rank_payees(_transaction("SUPA IGA NORTH MELB NORTH MELBOUVIC"))
        assert ranked.top is not None and ranked.top[0] == "Supa IGA"

    def test_very_short_payees_are_not_matched(self):
        """A payee called 'W' would otherwise match most of the corpus."""
        corpus = _corpus([], known_payees={"W"})
        suggester = Suggester(corpus)
        assert suggester.rank_payees(_transaction("W VARIETY STORE PTY LT")).top is None

    def test_the_account_follows_from_the_journal_prior(self):
        corpus = _corpus(
            [],
            known_payees={"City Beach"},
            known_accounts={"Expenses:Clothing"},
            payee_accounts={"City Beach": Counter({"Expenses:Clothing": 9})},
        )
        suggester = Suggester(corpus)
        transaction = _transaction("City Beach Highpoint Maribyrnong AUS")
        payee = suggester.rank_payees(transaction).top
        assert payee is not None
        accounts = suggester.rank_accounts(transaction, payee[0], OWN)
        assert accounts.top is not None and accounts.top[0] == "Expenses:Clothing"


class TestAccountLayer:
    def test_the_banks_own_account_is_never_suggested(self):
        corpus = _corpus(
            [_example("TRANSFER", "Me", OWN)],
            payee_accounts={"Me": Counter({OWN: 20, "Assets:Bank:Savings": 3})},
        )
        suggester = Suggester(corpus)
        accounts = suggester.rank_accounts(_transaction("TRANSFER"), "Me", OWN)
        assert all(account != OWN for account, _, _ in accounts.candidates)

    def test_the_confirmed_payee_narrows_a_shared_description(self):
        description = "PAYPAL 4029357733"
        corpus = _corpus(
            [
                _example(description, "Uber", "Expenses:Transport"),
                _example(description, "Spotify", "Expenses:Bills:Music"),
            ]
        )
        suggester = Suggester(corpus)
        accounts = suggester.rank_accounts(_transaction(description), "Spotify", OWN)
        assert accounts.top is not None and accounts.top[0] == "Expenses:Bills:Music"


class TestHoldout:
    def test_hides_the_row_being_asked_about(self):
        description = "WW METRO 8281 MELBOURNE VI"
        corpus = _corpus([_example(description, "Woolworths", "Expenses:Food")])
        suggester = Suggester(corpus)
        assert suggester.rank_payees(_transaction(description), holdout=True).top is None


class TestIncrementalLearning:
    def test_a_confirmation_is_usable_immediately(self):
        """No refit: the next transaction in the same session already benefits."""
        suggester = Suggester(_corpus([]))
        transaction = _transaction("NEW MERCHANT PTY LTD")
        assert suggester.rank_payees(transaction).top is None

        suggester.add(
            _example("NEW MERCHANT PTY LTD", "New Merchant", "Expenses:Other", origin="confirmed")
        )
        assert suggester.rank_payees(transaction).top == ("New Merchant", 1.0, "exact")

    def test_the_vector_index_grows_with_it(self):
        suggester = Suggester(_corpus([_example("A MERCHANT", "A", "Expenses:A")]))
        before = suggester.matrix.shape[0]
        suggester.add(_example("B MERCHANT", "B", "Expenses:B"))
        assert suggester.matrix.shape[0] == before + 1

    def test_a_near_variant_of_a_confirmation_is_matched(self):
        suggester = Suggester(_corpus([_example("A MERCHANT", "A", "Expenses:A")]))
        suggester.add(_example("KOKO BLACK 4821 CARLTON VIC", "Koko Black", "Expenses:Food"))
        ranked = suggester.rank_payees(_transaction("KOKO BLACK 9999 FITZROY VIC"))
        assert ranked.top is not None and ranked.top[0] == "Koko Black"
