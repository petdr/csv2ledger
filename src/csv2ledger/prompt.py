"""The interactive layer: show a transaction, confirm or correct the suggestion.

The design target is a run of Enter presses, so everything here is arranged around making
Enter mean "yes". The suggestion is pre-inserted into the readline buffer rather than
printed as a default in brackets: pressing Enter accepts it, and editing it is ordinary
line editing rather than retyping the whole payee.

Colon-prefixed commands (``:s``, ``:q``, ``:2``) are used instead of bare letters because
the buffer already contains a payee, and a bare ``s`` is a plausible payee in its own
right.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from .model import Transaction
from .suggest import Ranked, Suggester

try:  # readline is absent on some platforms; the prompts still work without it
    import readline
except ImportError:  # pragma: no cover - platform dependent
    readline = None  # type: ignore[assignment]

# For --auto: a high top score with a close runner-up is "confident but ambiguous" and
# still deserves a prompt.
AUTO_MARGIN = 0.10

HELP = """  Enter accepts the suggestion.  Tab completes.
  :s skip this transaction   :q stop here   :? show alternatives   :2 pick candidate 2
"""


@dataclass
class Decision:
    """What the user decided about one transaction."""

    payee: str = ""
    account: str = ""
    # True only when both fields were taken exactly as suggested -- the measure of
    # whether the tool is achieving its goal.
    accepted: bool = False
    skipped: bool = False
    quit: bool = False
    auto: bool = False
    suggested_payee: str = ""
    suggested_account: str = ""
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)


class _Completer:
    """Prefix completion over a fixed candidate list.

    Delimiters are cleared so the whole line is the prefix: payees contain spaces and
    accounts contain colons, and with default delimiters readline would try to complete
    only the fragment after the last space.
    """

    def __init__(self) -> None:
        self.options: Sequence[str] = ()
        self._matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        if state == 0:
            lowered = text.lower()
            self._matches = sorted(o for o in self.options if o.lower().startswith(lowered))
            if not self._matches:
                self._matches = sorted(o for o in self.options if lowered in o.lower())
        return self._matches[state] if state < len(self._matches) else None


class Prompter:
    """Drives the per-transaction dialogue."""

    def __init__(
        self,
        suggester: Suggester,
        own_account: str,
        default_account: str,
        *,
        auto_threshold: float | None = None,
        out=sys.stderr,
    ) -> None:
        self.suggester = suggester
        self.own_account = own_account
        self.default_account = default_account
        self.auto_threshold = auto_threshold
        self.out = out
        self._completer = _Completer()
        self._eof = False
        self.interactive = sys.stdin.isatty()
        self._stdout_is_tty = sys.stdout.isatty()
        if readline is not None:
            readline.set_completer(self._completer.complete)
            readline.set_completer_delims("")
            readline.parse_and_bind("tab: complete")

    # -- one transaction ------------------------------------------------------------

    def ask(
        self,
        transaction: Transaction,
        *,
        index: int = 0,
        total: int = 0,
        note: str = "",
    ) -> Decision:
        payees = self.suggester.rank_payees(transaction)
        suggested_payee = payees.top[0] if payees.top else transaction.description
        payee_confidence = payees.top[1] if payees.top else 0.0

        accounts = self.suggester.rank_accounts(transaction, suggested_payee, self.own_account)
        suggested_account = accounts.top[0] if accounts.top else self.default_account
        account_confidence = accounts.top[1] if accounts.top else 0.0

        confidence = min(payee_confidence, account_confidence)
        if self._auto_ok(payees, accounts, note):
            return Decision(
                payee=suggested_payee,
                account=suggested_account,
                accepted=True,
                auto=True,
                suggested_payee=suggested_payee,
                suggested_account=suggested_account,
                confidence=confidence,
            )

        self._show(transaction, index, total, payees, accounts, note)

        payee = self._field("Payee", suggested_payee, payees, self.suggester.corpus.known_payees)
        if payee is None:
            return Decision(skipped=True, suggested_payee=suggested_payee)
        if payee == "":
            return Decision(quit=True)

        # Re-rank the account against the payee the user actually settled on: correcting
        # the payee is meant to fix the account for free.
        if payee != suggested_payee:
            accounts = self.suggester.rank_accounts(transaction, payee, self.own_account)
            suggested_account = accounts.top[0] if accounts.top else self.default_account

        account = self._field(
            "Account", suggested_account, accounts, self.suggester.corpus.known_accounts
        )
        if account is None:
            return Decision(skipped=True, suggested_payee=suggested_payee)
        if account == "":
            return Decision(quit=True)

        return Decision(
            payee=payee,
            account=account,
            accepted=(payee == suggested_payee and account == suggested_account),
            suggested_payee=suggested_payee,
            suggested_account=suggested_account,
            confidence=confidence,
        )

    def _auto_ok(self, payees: Ranked, accounts: Ranked, note: str) -> bool:
        """Whether this row can be committed without asking.

        A flagged possible duplicate always prompts, however confident the model is:
        posting a transaction twice is a worse error than a misfiled payee.
        """
        if self.auto_threshold is None or note:
            return False
        if not payees.top or not accounts.top:
            return False
        return (
            payees.top[1] >= self.auto_threshold
            and accounts.top[1] >= self.auto_threshold
            and payees.margin >= AUTO_MARGIN
        )

    # -- display --------------------------------------------------------------------

    def _show(
        self,
        transaction: Transaction,
        index: int,
        total: int,
        payees: Ranked,
        accounts: Ranked,
        note: str,
    ) -> None:
        counter = f"[{index}/{total}] " if total else ""
        amount = transaction.credit or ("-" + transaction.debit if transaction.debit else "")
        print(file=self.out)
        print(f"{counter}{transaction.ledger_date}  {amount:>12}", file=self.out)
        print(f"  {' '.join(transaction.description.split())}", file=self.out)
        if note:
            print(f"  ! {note}", file=self.out)
        self._alternatives("payee", payees)
        self._alternatives("account", accounts)

    def _alternatives(self, label: str, ranked: Ranked) -> None:
        """List runners-up when the top candidate is not clearly ahead.

        Shown up front rather than behind ``:?`` so the common correction -- the right
        answer is second -- costs one keystroke instead of a round trip.
        """
        if not ranked.top or (ranked.top[1] >= 0.9 and ranked.margin >= AUTO_MARGIN):
            return
        rest = ranked.candidates[1:4]
        if not rest:
            return
        options = "  ".join(
            f":{position + 2} {value} ({score:.2f})"
            for position, (value, score, _) in enumerate(rest)
        )
        print(f"  {label:<8}{options}", file=self.out)

    # -- input ----------------------------------------------------------------------

    def _field(self, label: str, default: str, ranked: Ranked, options: set[str]) -> str | None:
        """Read one field. Returns the value, None to skip, or "" to quit."""
        while True:
            text = self._read(label, default, options)
            if text is None:  # EOF
                return default
            text = text.strip()

            if not text:
                return default
            if not text.startswith(":"):
                return text

            command = text[1:].strip().lower()
            if command in {"s", "skip"}:
                return None
            if command in {"q", "quit"}:
                return ""
            if command in {"?", "h", "help"}:
                self._show_candidates(ranked)
                continue
            if command.isdigit():
                position = int(command) - 1
                if 0 <= position < len(ranked.candidates):
                    return ranked.candidates[position][0]
                print(f"  no candidate {command}", file=self.out)
                continue
            print(f"  unknown command {text!r}", file=self.out)
            print(HELP, file=self.out, end="")

    def _show_candidates(self, ranked: Ranked) -> None:
        if not ranked.candidates:
            print("  (no candidates)", file=self.out)
            return
        for position, (value, score, source) in enumerate(ranked.candidates[:9], start=1):
            print(f"    :{position}  {value:<40} {score:.2f}  {source}", file=self.out)

    def _read(self, label: str, default: str, options: set[str]) -> str | None:
        if self._eof:
            return None
        self._completer.options = sorted(options)
        # Only pre-insert into a real terminal: with piped input there is no line buffer
        # to insert into, and an empty line is read as "accept" anyway.
        rl = readline if self.interactive else None
        if rl is not None:
            rl.set_startup_hook(lambda: rl.insert_text(default))
        try:
            if self._stdout_is_tty:
                return input(f"{label:<8}> ")
            # Entries are going to stdout, so the prompt must not: input() writes its
            # prompt there, which would land in the middle of the ledger output.
            print(f"{label:<8}> ", file=self.out, end="", flush=True)
            return input()
        except EOFError:
            # Piped input that has run out, or Ctrl-D. Accept the remaining suggestions
            # rather than aborting a part-finished import.
            self._eof = True
            print("\n  (end of input -- accepting suggestions for the rest)", file=self.out)
            return None
        finally:
            if rl is not None:
                rl.set_startup_hook()
