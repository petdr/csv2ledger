#!/usr/bin/env python3
"""Measure how well the suggester reproduces decisions the user actually made.

The corpus of hand-written mapping rules is split: the model trains on one part and is
asked to predict the labels of the other. Each held-out row is a real description the
user once had to classify by hand, so top-1 accuracy answers the question that matters --
"if this rule did not exist, would the suggestion still have been right?"

The rule layer is deliberately switched off. With it on, every held-out regex row would
be answered by the rule that produced it, and the score would measure nothing.

Usage:
    uv run python scripts/eval.py [--data DIR] [--holdout 0.2] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csv2ledger import seed as seed_module  # noqa: E402
from csv2ledger.model import Transaction  # noqa: E402
from csv2ledger.seed import SeedCorpus  # noqa: E402
from csv2ledger.suggest import Suggester  # noqa: E402

# Placeholder used when the held-out example gives no bank context.
OWN_ACCOUNT = "Assets:Bank:Placeholder"


def _fake_transaction(description: str) -> Transaction:
    from datetime import date

    return Transaction(
        bank="",
        ledger_date="2026/01/01",
        entry_date=date(2026, 1, 1),
        description=description,
        credit="",
        debit="1.00",
        raw_csv="",
        row_index=0,
    )


def evaluate(data_dir: Path, holdout: float, seed_value: int) -> int:
    corpus = seed_module.build(
        seed_module.all_mapping_files(data_dir / "mappings"),
        data_dir / "ross-family.dat",
    )
    for note in corpus.notes:
        print(f"  {note}")

    examples = list(corpus.examples)
    rng = random.Random(seed_value)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - holdout))
    train, test = examples[:split], examples[split:]
    print(f"\n  train={len(train)}  test={len(test)}\n")

    train_corpus = SeedCorpus(
        examples=train,
        payee_accounts=corpus.payee_accounts,
        known_payees=corpus.known_payees,
        known_accounts=corpus.known_accounts,
    )
    # rules=[] on purpose: see the module docstring.
    suggester = Suggester(train_corpus, rules=[])

    # A held-out row whose payee never appears in training cannot be predicted by any
    # model -- the label does not exist yet. 74% of payees in this corpus have exactly
    # one example (one-off merchants), so that ceiling is low and must be reported, or
    # the headline accuracy looks like a model failure when it is a data property.
    train_payees = {e.payee for e in train}
    train_accounts = {e.account for e in train}
    reachable_payee = sum(1 for e in test if e.payee in train_payees)
    reachable_account = sum(1 for e in test if e.account in train_accounts)

    payee_top1 = payee_top3 = 0
    account_top1 = account_top3 = 0
    account_top1_given_payee = 0
    payee_top1_reachable = 0
    account_top1_reachable = 0
    both = 0
    no_suggestion = 0
    buckets: Counter[str] = Counter()
    bucket_hits: Counter[str] = Counter()

    for example in test:
        transaction = _fake_transaction(example.description)

        payees = suggester.rank_payees(transaction)
        payee_names = [value for value, _, _ in payees.candidates[:3]]
        predicted_payee = payee_names[0] if payee_names else ""
        payee_ok = predicted_payee == example.payee
        payee_top1 += payee_ok
        payee_top3 += example.payee in payee_names

        # Account accuracy as the pipeline actually runs it: chained off the predicted
        # payee, so a payee mistake is allowed to cost an account mistake.
        accounts = suggester.rank_accounts(transaction, predicted_payee, OWN_ACCOUNT)
        account_names = [value for value, _, _ in accounts.candidates[:3]]
        account_ok = bool(account_names) and account_names[0] == example.account
        account_top1 += account_ok
        account_top3 += example.account in account_names

        # And again with the correct payee supplied, which is what the user sees after
        # they fix the payee -- it isolates the account model from payee errors.
        oracle = suggester.rank_accounts(transaction, example.payee, OWN_ACCOUNT)
        account_top1_given_payee += bool(oracle.candidates) and (
            oracle.candidates[0][0] == example.account
        )

        if example.payee in train_payees:
            payee_top1_reachable += payee_ok
        if example.account in train_accounts:
            account_top1_reachable += account_ok

        both += payee_ok and account_ok
        if not payee_names:
            no_suggestion += 1

        confidence = payees.candidates[0][1] if payees.candidates else 0.0
        bucket = _bucket(confidence)
        buckets[bucket] += 1
        bucket_hits[bucket] += payee_ok

    total = len(test)
    print(f"  {'payee top-1':<28} {_pct(payee_top1, total)}")
    print(f"  {'payee top-3':<28} {_pct(payee_top3, total)}")
    print(f"  {'account top-1 (chained)':<28} {_pct(account_top1, total)}")
    print(f"  {'account top-3 (chained)':<28} {_pct(account_top3, total)}")
    print(f"  {'account top-1 (payee given)':<28} {_pct(account_top1_given_payee, total)}")
    print(f"  {'both correct (pure Enter)':<28} {_pct(both, total)}")
    print(f"  {'no suggestion at all':<28} {_pct(no_suggestion, total)}")

    print("\n  against the achievable ceiling (label must exist in training):")
    print(f"    {'payee ceiling':<26} {_pct(reachable_payee, total)}")
    print(
        f"    {'payee top-1 of reachable':<26} "
        f"{_pct(payee_top1_reachable, reachable_payee)}"
    )
    print(f"    {'account ceiling':<26} {_pct(reachable_account, total)}")
    print(
        f"    {'account top-1 of reachable':<26} "
        f"{_pct(account_top1_reachable, reachable_account)}"
    )

    print("\n  payee accuracy by confidence:")
    print(f"    {'bucket':<12} {'n':>6} {'correct':>9}  {'precision':>9}")
    for bucket in sorted(buckets, reverse=True):
        n = buckets[bucket]
        hits = bucket_hits[bucket]
        print(f"    {bucket:<12} {n:>6} {hits:>9}  {hits / n:>8.1%}")

    return 0


def _bucket(confidence: float) -> str:
    for edge in (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30):
        if confidence >= edge:
            return f">={edge:.2f}"
    return "<0.30"


def _pct(count: int, total: int) -> str:
    return f"{count:>6} / {total}  {count / total:>7.1%}" if total else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path.home() / "Dropbox" / "financial")
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return evaluate(args.data, args.holdout, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
