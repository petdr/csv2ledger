#!/usr/bin/env python3
"""Measure the suggester on a real bank CSV, as if every description were new.

scripts/eval.py splits the mapping corpus uniformly, which over-weights one-off
merchants: three quarters of the payees in that corpus occur exactly once, so most of
what it tests is unrepeatable by construction. A real download is weighted the way real
spending is -- the same dozen merchants over and over -- and that is the distribution the
"just press Enter" goal is actually about.

Ground truth is the mapping rule the old tool would have used, which is the decision the
user made when they first classified that description. To keep the question honest the
model is run with:

  * the bank's rule layer switched off, and
  * every training example whose normalised description matches the row hidden

so it has to answer from *similar* transactions rather than from the row itself.

Remaining leak, stated rather than hidden: a regex mapping row contributes a
de-metacharacterised training string that the holdout cannot recognise as the same row,
so a description first classified by regex can still be helped by its own rule.

Usage:
    uv run python scripts/realcheck.py -c /tmp/banks.toml -a COMMBANK ~/Dropbox/financial/COMMBANK.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csv2ledger import config as config_module  # noqa: E402
from csv2ledger import csvsource, mappings, seed  # noqa: E402
from csv2ledger.prompt import AUTO_MARGIN  # noqa: E402
from csv2ledger.suggest import Suggester  # noqa: E402

THRESHOLDS = (0.99, 0.95, 0.90, 0.85, 0.80)


def check(config: Path, account: str, csvfile: Path) -> int:
    banks = config_module.load(config)
    bank = banks[account]

    rules = mappings.read_all([bank.legacy_mapping_file]).rules if bank.legacy_mapping_file else []
    transactions = list(csvsource.read(csvfile, bank))

    mapping_dir = bank.mappings_dir or (
        bank.legacy_mapping_file.parent if bank.legacy_mapping_file else None
    )
    corpus = seed.build(
        seed.all_mapping_files(mapping_dir) if mapping_dir else [], bank.journal
    )
    for note in corpus.notes:
        print(f"  {note}")
    # rules=[] -- the rule layer would otherwise answer every row from the very rule
    # being used as ground truth.
    suggester = Suggester(corpus, rules=[])

    counts: Counter[str] = Counter()
    misses: list[tuple[str, str, str, str, str]] = []

    for transaction in transactions:
        rule = mappings.match(rules, transaction.description)
        if rule is None:
            counts["unlabelled"] += 1
            continue
        counts["labelled"] += 1

        payees = suggester.rank_payees(transaction, holdout=True)
        payee = payees.top[0] if payees.top else ""
        confidence = payees.top[1] if payees.top else 0.0

        accounts = suggester.rank_accounts(
            transaction, payee, bank.account, holdout=True
        )
        predicted = accounts.top[0] if accounts.top else ""

        payee_ok = payee == rule.payee
        account_ok = predicted == rule.account
        counts["payee_ok"] += payee_ok
        counts["account_ok"] += account_ok
        counts["both_ok"] += payee_ok and account_ok
        # Exactly the gate Prompter applies, so the sweep reports what --auto would do
        # rather than something close to it.
        account_confidence = accounts.top[1] if accounts.top else 0.0
        for threshold in THRESHOLDS:
            if (
                confidence >= threshold
                and account_confidence >= threshold
                and payees.margin >= AUTO_MARGIN
            ):
                counts[f"auto_{threshold}"] += 1
                counts[f"hit_{threshold}"] += payee_ok and account_ok
        if not (payee_ok and account_ok):
            misses.append(
                (transaction.description, rule.payee, payee, rule.account, predicted)
            )

    labelled = counts["labelled"]
    print(f"\n  {len(transactions)} rows, {labelled} with a mapping rule to check against")
    if counts["unlabelled"]:
        print(f"  {counts['unlabelled']} rows have no rule (no ground truth, not scored)")
    if not labelled:
        return 0

    print(f"\n  {'payee correct':<28} {_pct(counts['payee_ok'], labelled)}")
    print(f"  {'account correct':<28} {_pct(counts['account_ok'], labelled)}")
    print(f"  {'both correct (pure Enter)':<28} {_pct(counts['both_ok'], labelled)}")
    # What --auto would actually do: how much of the file it takes off your hands, and
    # how often it is wrong when it does. A wrong auto-commit is silent, so the precision
    # column is the one that sets the threshold.
    print("\n  --auto threshold sweep:")
    print(f"    {'threshold':<10} {'unattended':>11} {'correct':>9} {'precision':>10}")
    for threshold in THRESHOLDS:
        eligible = counts[f"auto_{threshold}"]
        if not eligible:
            continue
        hits = counts[f"hit_{threshold}"]
        print(
            f"    {threshold:<10} {eligible:>11} {hits:>9} {hits / eligible:>9.1%}"
            f"   ({eligible / labelled:.0%} of file)"
        )

    print(f"\n  misses ({len(misses)}):")
    for description, want_payee, got_payee, want_account, got_account in misses[:40]:
        print(f"    {' '.join(description.split())[:52]:<52}")
        if want_payee != got_payee:
            print(f"      payee    want {want_payee!r:<32} got {got_payee!r}")
        if want_account != got_account:
            print(f"      account  want {want_account!r:<32} got {got_account!r}")
    if len(misses) > 40:
        print(f"    ... and {len(misses) - 40} more")
    return 0


def _pct(count: int, total: int) -> str:
    return f"{count:>5} / {total}  {count / total:>7.1%}" if total else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-a", "--account", required=True)
    parser.add_argument("csvfile", type=Path)
    args = parser.parse_args()
    return check(args.config, args.account, args.csvfile)


if __name__ == "__main__":
    raise SystemExit(main())
