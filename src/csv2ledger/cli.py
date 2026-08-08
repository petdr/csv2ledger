"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from . import config as config_module
from . import csvsource, dedupe, mappings, migrate, render, seed, store
from .normalize import normalize
from .prompt import Prompter
from .suggest import Suggester

DEFAULT_CONFIG = Path.home() / ".config" / "csv2ledger" / "banks.toml"
DEFAULT_DATA = Path.home() / ".local" / "share" / "csv2ledger"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"path to banks.toml (default: {DEFAULT_CONFIG})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csv2ledger",
        description="Import bank CSVs into ledger format, learning payees and accounts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    migrate_cmd = sub.add_parser("migrate", help="convert an .icsv2ledgerrc into banks.toml")
    migrate_cmd.add_argument("rcfile", type=Path, help="path to the existing .icsv2ledgerrc")
    migrate_cmd.add_argument(
        "-o", "--output", type=Path, default=None, help="write to this file instead of stdout"
    )
    migrate_cmd.add_argument(
        "--root",
        type=Path,
        default=None,
        help="directory that relative paths resolve against (default: the rcfile's directory)",
    )

    import_cmd = sub.add_parser("import", help="convert a bank CSV into ledger entries")
    _add_common(import_cmd)
    import_cmd.add_argument("-a", "--account", required=True, help="bank name from banks.toml")
    import_cmd.add_argument("csvfile", type=Path, help="the bank-supplied CSV")
    import_cmd.add_argument(
        "-o", "--output", type=Path, default=None, help="write entries here (default: stdout)"
    )
    import_cmd.add_argument(
        "--skip-older-than",
        type=int,
        default=-1,
        metavar="DAYS",
        help="ignore transactions older than DAYS (-1 disables, the default)",
    )
    import_cmd.add_argument(
        "--rules-only",
        action="store_true",
        help="use only the legacy mapping rules, no model and no prompting "
        "(this is the icsv2ledger-compatible mode used by the differential test)",
    )
    import_cmd.add_argument(
        "--auto",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="commit suggestions at or above this confidence without asking "
        "(e.g. --auto 0.95); everything else still prompts",
    )
    import_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="do not write entries or record anything learned; just report what would happen",
    )
    import_cmd.add_argument(
        "--no-learn",
        action="store_true",
        help="do not append confirmations to the training store",
    )
    import_cmd.add_argument(
        "--no-dedupe",
        action="store_true",
        help="do not skip rows already imported (fingerprint check)",
    )

    banks_cmd = sub.add_parser("banks", help="list configured banks")
    _add_common(banks_cmd)

    stats_cmd = sub.add_parser("stats", help="summarise the training store")
    _add_common(stats_cmd)

    retrain_cmd = sub.add_parser(
        "retrain", help="compact the training store and rebuild the index"
    )
    _add_common(retrain_cmd)
    retrain_cmd.add_argument(
        "-a",
        "--account",
        default=None,
        help="bank whose paths locate the store (default: the first in banks.toml)",
    )
    retrain_cmd.add_argument(
        "--no-compact",
        action="store_true",
        help="rebuild and report only; leave the training store exactly as it is",
    )

    return parser


def cmd_migrate(args: argparse.Namespace) -> int:
    text, warnings = migrate.convert(args.rcfile, root=args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def cmd_banks(args: argparse.Namespace) -> int:
    banks = config_module.load(args.config)
    width = max(len(name) for name in banks)
    for name in sorted(banks):
        bank = banks[name]
        print(f"{name:<{width}}  {bank.account}")
    return 0


def _load_rules(bank: config_module.BankConfig) -> list[mappings.Rule]:
    """The legacy rules for one bank, reporting anything unparseable."""
    if not bank.legacy_mapping_file:
        return []
    report = mappings.read_all([bank.legacy_mapping_file])
    for bad in report.skipped_bad_regex:
        print(f"warning: skipping invalid regex -- {bad}", file=sys.stderr)
    if report.skipped_short:
        print(
            f"warning: skipped {report.skipped_short} malformed mapping row(s) in "
            f"{bank.legacy_mapping_file.name}",
            file=sys.stderr,
        )
    return report.rules


def _data_path(configured: Path | None, filename: str) -> Path:
    return configured if configured is not None else DEFAULT_DATA / filename


def _build_suggester(bank: config_module.BankConfig, rules: list[mappings.Rule]) -> Suggester:
    """Seed corpus + everything confirmed so far, as one index.

    Every bank's mapping file is read, not just this one's: the sharing is the point.
    """
    if bank.mappings_dir is not None:
        mapping_paths = seed.all_mapping_files(bank.mappings_dir)
    elif bank.legacy_mapping_file is not None:
        mapping_paths = [bank.legacy_mapping_file]
    else:
        mapping_paths = []

    corpus = seed.build(mapping_paths, bank.journal)

    training = store.TrainingStore(_data_path(bank.training_file, "training.jsonl"))
    confirmations = training.load()
    for confirmation in confirmations:
        corpus.examples.append(confirmation.to_example())
        corpus.known_payees.add(confirmation.payee)
        corpus.known_accounts.add(confirmation.account)
        corpus.payee_accounts[confirmation.payee][confirmation.account] += 1
    if confirmations:
        corpus.notes.append(f"training store: {len(confirmations)} confirmation(s)")

    for note in corpus.notes:
        print(f"  {note}", file=sys.stderr)
    return Suggester(corpus, rules)


def cmd_import(args: argparse.Namespace) -> int:
    banks = config_module.load(args.config)
    if args.account not in banks:
        print(
            f"error: unknown bank {args.account!r}; known banks: {', '.join(sorted(banks))}",
            file=sys.stderr,
        )
        return 2
    bank = banks[args.account]
    rules = _load_rules(bank)

    template = bank.template()
    skip = args.skip_older_than if args.skip_older_than >= 0 else None
    transactions = list(csvsource.read(args.csvfile, bank, skip_older_than=skip))

    if args.rules_only:
        entries = [
            _rules_only_entry(transaction, rules, bank, template)
            for transaction in transactions
        ]
        _write(entries, args.output)
        return 0

    suggester = _build_suggester(bank, rules)
    prompter = Prompter(
        suggester,
        own_account=bank.account,
        default_account=bank.default_account,
        auto_threshold=args.auto,
    )
    training = store.TrainingStore(_data_path(bank.training_file, "training.jsonl"))
    fingerprints = dedupe.FingerprintLog(_data_path(bank.fingerprint_file, "imported.log"))
    journal = dedupe.journal_index(bank.journal) if not args.no_dedupe else Counter()

    learn = not (args.dry_run or args.no_learn)
    entries: list[str] = []
    counts = Counter[str]()

    for position, transaction in enumerate(transactions, start=1):
        if not args.no_dedupe and fingerprints.seen(transaction):
            counts["duplicate"] += 1
            continue

        note = ""
        if journal[dedupe.journal_key(transaction)]:
            note = "a transaction with this date and amount is already in the journal"

        try:
            decision = prompter.ask(
                transaction, index=position, total=len(transactions), note=note
            )
        except KeyboardInterrupt:
            # Keep what has been classified so far rather than discarding the session.
            print("\ninterrupted", file=sys.stderr)
            counts["unprocessed"] = len(transactions) - position + 1
            break
        if decision.quit:
            counts["unprocessed"] = len(transactions) - position + 1
            break
        if decision.skipped:
            counts["skipped"] += 1
            continue

        counts["written"] += 1
        counts["accepted"] += decision.accepted
        counts["auto"] += decision.auto
        counts["payee_overridden"] += decision.payee != decision.suggested_payee
        counts["account_overridden"] += decision.account != decision.suggested_account

        rule = mappings.match(rules, transaction.description)
        tags = rule.tags if rule is not None and rule.payee == decision.payee else []
        entries.append(
            render.journal_entry(
                transaction,
                decision.payee,
                decision.account,
                bank,
                template=template,
                tags=tags,
            )
        )
        confirmation = store.Confirmation(
            description=transaction.description,
            normalized=normalize(transaction.description),
            payee=decision.payee,
            account=decision.account,
            bank=bank.name,
            direction=transaction.direction,
            at=store.now_stamp(),
            overridden=not decision.accepted,
            suggested_payee=decision.suggested_payee,
            suggested_account=decision.suggested_account,
            confidence=decision.confidence,
            tags=tags,
        )
        # Fold the confirmation into the live index either way: within a single import a
        # description usually recurs, and the second occurrence should not have to be
        # answered again even on a dry run.
        suggester.add(confirmation.to_example())
        if learn:
            training.append(confirmation)
            fingerprints.record(transaction, decision.payee)
        else:
            # Still count it against this run, so a row occurring twice in the file is
            # not mistaken for the earlier import of its first occurrence.
            fingerprints.mark(transaction)

    _write(entries, None if args.dry_run else args.output)
    _report(counts, len(transactions), args.dry_run)
    return 0


def _rules_only_entry(
    transaction, rules: list[mappings.Rule], bank: config_module.BankConfig, template: str
) -> str:
    rule = mappings.match(rules, transaction.description)
    if rule is not None:
        payee, account, tags = rule.payee, rule.account, rule.tags
    else:
        payee, account, tags = transaction.description, bank.default_account, []
    return render.journal_entry(transaction, payee, account, bank, template=template, tags=tags)


def _write(entries: list[str], output: Path | None) -> None:
    if output:
        with output.open("w", encoding="utf-8") as handle:
            render.write_entries(entries, handle)
    else:
        render.write_entries(entries, sys.stdout)


def _report(counts: Counter[str], total: int, dry_run: bool) -> None:
    written = counts["written"]
    print(f"\n{'(dry run) ' if dry_run else ''}{total} row(s) read", file=sys.stderr)
    if counts["duplicate"]:
        print(f"  {counts['duplicate']:>6} already imported, skipped", file=sys.stderr)
    if counts["skipped"]:
        print(f"  {counts['skipped']:>6} skipped by you", file=sys.stderr)
    if counts["unprocessed"]:
        print(f"  {counts['unprocessed']:>6} left unprocessed (you quit)", file=sys.stderr)
    print(f"  {written:>6} entries written", file=sys.stderr)
    if not written:
        return
    share = counts["accepted"] / written
    automatic = f", {counts['auto']} of them automatically" if counts["auto"] else ""
    print(
        f"  {counts['accepted']:>6} accepted as suggested ({share:.1%}){automatic}",
        file=sys.stderr,
    )
    print(f"  {counts['payee_overridden']:>6} payee corrections", file=sys.stderr)
    print(f"  {counts['account_overridden']:>6} account corrections", file=sys.stderr)


def cmd_stats(args: argparse.Namespace) -> int:
    banks = config_module.load(args.config)
    bank = next(iter(banks.values()))
    training = store.TrainingStore(_data_path(bank.training_file, "training.jsonl"))
    confirmations = training.load()
    if not confirmations:
        print(f"no confirmations yet at {training.path}", file=sys.stderr)
        return 0

    overridden = sum(1 for c in confirmations if c.overridden)
    payees = {c.payee for c in confirmations}
    accounts = {c.account for c in confirmations}
    print(f"{training.path}")
    print(f"  {len(confirmations)} confirmation(s), {len(payees)} payees, {len(accounts)} accounts")
    print(f"  {overridden} overridden ({overridden / len(confirmations):.1%})")
    by_bank = Counter(c.bank for c in confirmations)
    for name, count in by_bank.most_common():
        print(f"    {name or '(unknown)':<16} {count}")
    return 0


# A token appearing in this share of confirmations but almost never in the original
# mapping corpus is a description-format change rather than a new merchant: merchants come
# and go one at a time, boilerplate lands on everything at once.
DRIFT_MIN_SHARE = 0.20
DRIFT_MAX_SEED_SHARE = 0.02
# Below this many confirmations the shares are noise.
DRIFT_MIN_CONFIRMATIONS = 50


def _token_shares(descriptions: list[str]) -> dict[str, float]:
    """Fraction of descriptions each token appears in."""
    if not descriptions:
        return {}
    counts: Counter[str] = Counter()
    for description in descriptions:
        counts.update(set(description.split()))
    return {token: count / len(descriptions) for token, count in counts.items()}


def _drift(suggester: Suggester) -> list[tuple[str, float, float]]:
    """Tokens that boilerplate recent confirmations but were unknown to the seed corpus."""
    seeded = [e.description for e in suggester.examples if e.origin.startswith("mapping")]
    confirmed = [e.description for e in suggester.examples if e.origin == "confirmed"]
    if len(confirmed) < DRIFT_MIN_CONFIRMATIONS:
        return []

    seed_shares = _token_shares(seeded)
    return sorted(
        (
            (token, share, seed_shares.get(token, 0.0))
            for token, share in _token_shares(confirmed).items()
            if share >= DRIFT_MIN_SHARE and seed_shares.get(token, 0.0) <= DRIFT_MAX_SEED_SHARE
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def cmd_retrain(args: argparse.Namespace) -> int:
    """Housekeeping: collapse repeated confirmations, rebuild the index, report drift.

    Note what this deliberately does *not* do. IDF weights are refitted on every import
    anyway, because `_build_suggester` folds the training store into the corpus before the
    Suggester is constructed -- so there is no stale-weights problem for this command to
    solve, and claiming otherwise would be theatre. What it does solve is a training store
    that has accumulated thousands of repeats of the same decision.
    """
    banks = config_module.load(args.config)
    if args.account is not None and args.account not in banks:
        print(
            f"error: unknown bank {args.account!r}; known banks: {', '.join(sorted(banks))}",
            file=sys.stderr,
        )
        return 2
    bank = banks[args.account] if args.account else next(iter(banks.values()))

    training = store.TrainingStore(_data_path(bank.training_file, "training.jsonl"))
    before = len(training.load())
    if args.no_compact:
        summary = f"{before} confirmation(s), left untouched"
    elif not before:
        summary = "nothing recorded yet"
    else:
        summary = f"{before} confirmation(s), {training.compact()} duplicate(s) removed"
    # Flushed because the corpus notes below are written to stderr, and a reader piping
    # both streams together should still see these lines in the order they happened.
    print(f"{training.path}: {summary}", flush=True)

    started = time.perf_counter()
    suggester = _build_suggester(bank, _load_rules(bank))
    elapsed = time.perf_counter() - started
    rows, features = suggester.matrix.shape
    print(f"index: {rows} rows x {features} features, rebuilt in {elapsed:.1f}s")

    drift = _drift(suggester)
    if drift:
        print("\npossible description-format change -- these are common in your")
        print("confirmations but were absent from the mapping corpus:")
        for token, share, seed_share in drift[:10]:
            print(f"  {token:<24} {share:>6.1%} of confirmations, {seed_share:>5.1%} of mappings")
        print("\nThe index just rebuilt has already absorbed them, so this is a note about")
        print("the import where they first appeared, not about the next one.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "migrate": cmd_migrate,
        "import": cmd_import,
        "banks": cmd_banks,
        "stats": cmd_stats,
        "retrain": cmd_retrain,
    }
    try:
        return handlers[args.command](args)
    except (config_module.ConfigError, csvsource.CsvError, migrate.MigrationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
