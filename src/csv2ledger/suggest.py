"""Rank payee and account candidates for a transaction.

Four layers, in descending order of trust:

  1. exact match on the normalised description -- something identical was labelled before
  2. a regex rule from the legacy mapping files -- an explicit decision the user wrote
  3. nearest-neighbour over character n-grams -- the layer that generalises
  4. the payee-to-account prior from the journal -- account stage only

Nearest-neighbour is the workhorse rather than a trained linear classifier because of
the shape of this data: 3,415 payees over 6,167 examples means most payees have one or
two examples. A linear model has almost nothing to fit per class, whereas a neighbour
lookup degrades gracefully to "the closest thing you have ever labelled".

Prediction is chained: description to payee, then description plus the *confirmed* payee
to account. Chaining means correcting a wrong payee also corrects the account.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import mappings
from .features import FeatureSpace, cosine_scores
from .model import Example, Suggestion, Transaction
from .normalize import normalize
from .seed import SeedCorpus

# Neighbours below this similarity are noise rather than weak evidence.
MIN_SIMILARITY = 0.30
NEIGHBOURS = 8

# A known payee spelled out at the start of the description. Short names are excluded
# because a two- or three-letter payee ('W', 'BP') matches far too much.
MIN_LITERAL_LENGTH = 4
MAX_LITERAL_TOKENS = 6
# Scored between the neighbour layer's typical range and the rule layer: seeing the
# merchant's name written out is better evidence than a fuzzy match, but it is still a
# guess about which of several similarly-named payees is meant.
LITERAL_BASE = 0.70
LITERAL_PER_TOKEN = 0.04


def _vote(examples: list[Example], label, source: str) -> list[tuple[str, float, str]]:
    """Rank the labels of a set of exactly-matching examples by weighted frequency.

    The historical data is not self-consistent: the same description has been filed
    under 'Woolies' and 'Woolworths', 'Zouki' and 'Zoukia', over more than a decade.
    Taking the most frequent label rather than the most recent picks the spelling the
    user has actually settled on, and the returned confidence is the winner's share of
    the vote -- so a genuinely split description reports as uncertain rather than
    claiming 1.0 and being wrong half the time.
    """
    tally: dict[str, float] = defaultdict(float)
    for example in examples:
        tally[label(example)] += example.weight
    total = sum(tally.values())
    if not total:
        return []
    ordered = sorted(tally.items(), key=lambda item: item[1], reverse=True)
    return [(value, weight / total, source) for value, weight in ordered]


def _merge(
    candidates: list[tuple[str, float, str]], value: str, score: float, source: str
) -> list[tuple[str, float, str]]:
    """Fold one extra candidate into a ranked list, keeping the better score.

    Two layers agreeing is not treated as twice the evidence -- they are reading the same
    description -- so the scores are combined by max rather than added.

    The incoming list is *not* re-sorted. Neighbour candidates are ordered by their
    weighted vote while the score reported alongside them is the best single similarity,
    and those two orderings disagree; re-sorting would silently discard the vote.
    """
    existing = next((c for c in candidates if c[0] == value), None)
    rest = [c for c in candidates if c[0] != value]
    if existing is not None:
        score = max(score, existing[1])
        source = f"{existing[2]}+{source}"
    position = next(
        (i for i, (_, other, _) in enumerate(rest) if score > other), len(rest)
    )
    rest.insert(position, (value, score, source))
    return rest


@dataclass
class Ranked:
    """A ranked candidate list plus the confidence of the top entry."""

    candidates: list[tuple[str, float, str]]  # (value, score, source)

    @property
    def top(self) -> tuple[str, float, str] | None:
        return self.candidates[0] if self.candidates else None

    @property
    def margin(self) -> float:
        """Gap between the best and second-best candidate.

        A high top score with a small margin means "confident but ambiguous", which
        should still be shown to the user rather than auto-accepted.
        """
        if len(self.candidates) < 2:
            return self.candidates[0][1] if self.candidates else 0.0
        return self.candidates[0][1] - self.candidates[1][1]


class Suggester:
    """Holds the corpus, the vector index and the rule layer."""

    def __init__(
        self,
        corpus: SeedCorpus,
        rules: list[mappings.Rule] | None = None,
        space: FeatureSpace | None = None,
    ) -> None:
        self.corpus = corpus
        self.rules = rules or []
        self.space = space or FeatureSpace()
        self.examples: list[Example] = list(corpus.examples)
        self._matrix: sparse.csr_matrix | None = None
        self._pending: list[sparse.csr_matrix] = []
        # normalised description -> the labels seen for it, most recent last
        self._exact: dict[str, list[Example]] = defaultdict(list)
        for example in self.examples:
            self._exact[example.description].append(example)
        self._literal = self._build_literal_index()
        self._build()

    def _build_literal_index(self) -> dict[str, str]:
        """Map every known payee's normalised name back to its canonical spelling.

        Bank descriptions usually begin with the merchant name, so a payee already in the
        vocabulary can often be read straight off the front of a description that has
        never been seen before -- which is exactly the case nearest-neighbour is weakest
        at. Where two spellings normalise to the same string, the one used in more
        journal transactions wins.
        """
        index: dict[str, str] = {}
        ranking: dict[str, int] = {}
        for payee in self.corpus.known_payees:
            key = normalize(payee)
            if len(key) < MIN_LITERAL_LENGTH or len(key.split()) > MAX_LITERAL_TOKENS:
                continue
            uses = sum(self.corpus.payee_accounts.get(payee, {}).values())
            if key not in index or uses > ranking[key]:
                index[key] = payee
                ranking[key] = uses
        return index

    def _literal_match(self, key: str) -> tuple[str, float] | None:
        """Longest known payee spelled out at the start of ``key``, if any."""
        tokens = key.split()
        for length in range(min(MAX_LITERAL_TOKENS, len(tokens)), 0, -1):
            candidate = " ".join(tokens[:length])
            if len(candidate) < MIN_LITERAL_LENGTH:
                break
            payee = self._literal.get(candidate)
            if payee is not None:
                return payee, min(LITERAL_BASE + LITERAL_PER_TOKEN * length, 0.9)
        return None

    def _build(self) -> None:
        descriptions = [e.description for e in self.examples]
        banks = [e.bank for e in self.examples]
        directions = [e.direction for e in self.examples]
        # IDF is fitted once here and then held fixed, so rows appended later stay
        # comparable with rows already in the index.
        self.space.fit(descriptions, banks, directions)
        self._matrix = self.space.transform(descriptions, banks, directions)
        self._pending = []

    @property
    def matrix(self) -> sparse.csr_matrix:
        """The index, folding in any rows appended since the last query."""
        if self._pending:
            assert self._matrix is not None
            self._matrix = sparse.vstack([self._matrix, *self._pending], format="csr")
            self._pending = []
        assert self._matrix is not None
        return self._matrix

    def add(self, example: Example) -> None:
        """Append a confirmed example so the very next lookup can use it.

        Only one row is vectorised and stacked; nothing is refitted. This is why the
        feature space had to be hashing-based.
        """
        self.examples.append(example)
        self._exact[example.description].append(example)
        self._pending.append(
            self.space.transform_one(example.description, example.bank, example.direction)
        )
        self.corpus.known_payees.add(example.payee)
        self.corpus.known_accounts.add(example.account)

    # -- payee ------------------------------------------------------------------

    def rank_payees(self, transaction: Transaction, *, holdout: bool = False) -> Ranked:
        """Rank payees for a transaction.

        ``holdout`` hides everything the corpus already knows about this exact
        description, answering "what would you have suggested if you had never seen this
        before?". Evaluation needs it; normal use never sets it.
        """
        key = normalize(transaction.description)

        exact = None if holdout else self._exact.get(key)
        if exact:
            return Ranked(_vote(exact, lambda e: e.payee, "exact"))

        rule = mappings.match(self.rules, transaction.description)
        if rule is not None:
            return Ranked([(rule.payee, 0.99, "rule")])

        scored = self._neighbour_votes(key, transaction, lambda e: e.payee, holdout=holdout)
        literal = self._literal_match(key)
        if literal is not None:
            scored = _merge(scored, literal[0], literal[1], "literal")
        return Ranked(scored)

    # -- account ----------------------------------------------------------------

    def rank_accounts(
        self, transaction: Transaction, payee: str, own_account: str, *, holdout: bool = False
    ) -> Ranked:
        """Rank accounts for a transaction whose payee is now settled.

        ``own_account`` is the bank's own side of the entry; it is always one of the two
        postings and must never be suggested as the other one.
        """
        key = normalize(transaction.description)
        exclude = {own_account}

        exact = None if holdout else self._exact.get(key)
        if exact:
            # Prefer examples that also agree on the payee: once the payee is settled,
            # rows for a different payee are describing a different transaction.
            relevant = [e for e in exact if e.payee == payee] or exact
            voted = [
                (account, score, source)
                for account, score, source in _vote(relevant, lambda e: e.account, "exact")
                if account not in exclude
            ]
            if voted:
                return Ranked(voted)

        rule = mappings.match(self.rules, transaction.description)
        if rule is not None and rule.payee == payee and rule.account not in exclude:
            return Ranked([(rule.account, 0.99, "rule")])

        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, str] = {}

        for account, score, source in self._neighbour_votes(
            key, transaction, lambda e: e.account, holdout=holdout
        ):
            if account in exclude:
                continue
            scores[account] += score
            sources[account] = source

        # The journal prior: what account has this payee actually been posted to?
        # Strong evidence once the payee is known, and it covers payees whose
        # descriptions never appeared in a mapping file.
        prior = self.corpus.payee_accounts.get(payee)
        if prior:
            total = sum(count for acct, count in prior.items() if acct not in exclude)
            if total:
                for account, count in prior.items():
                    if account in exclude:
                        continue
                    weight = count / total
                    scores[account] += weight
                    sources.setdefault(account, "prior")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # Two independent signals can each contribute up to 1.0; rescale so the reported
        # confidence stays comparable with the exact/rule layers.
        return Ranked([(acct, min(score / 2.0, 0.98), sources[acct]) for acct, score in ranked])

    # -- shared -----------------------------------------------------------------

    def _neighbour_votes(
        self, key: str, transaction: Transaction, label, *, holdout: bool = False
    ) -> list[tuple[str, float, str]]:
        """Similarity-weighted vote over the nearest neighbours of ``key``."""
        if not key or self.matrix.shape[0] == 0:
            return []

        query = self.space.transform_one(key, transaction.bank, transaction.direction)
        similarities = cosine_scores(query, self.matrix)
        if similarities.size == 0:
            return []

        # Ask for extra neighbours under holdout, since some of the closest will be the
        # identical rows that are about to be discarded.
        count = min(NEIGHBOURS * (4 if holdout else 1), similarities.size)
        top = np.argpartition(-similarities, count - 1)[:count]
        top = top[np.argsort(-similarities[top])]

        votes: dict[str, float] = defaultdict(float)
        best: dict[str, float] = {}
        kept = 0
        for index in top:
            similarity = float(similarities[index])
            if similarity < MIN_SIMILARITY or kept >= NEIGHBOURS:
                break
            example = self.examples[index]
            if holdout and example.description == key:
                continue
            kept += 1
            value = label(example)
            votes[value] += similarity * example.weight
            best[value] = max(best.get(value, 0.0), similarity)

        if not votes:
            return []

        # Report the best single-neighbour similarity rather than the summed vote: the
        # sum grows with how many neighbours happen to agree, which would inflate
        # confidence for common merchants regardless of how good the match actually is.
        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        return [(value, best[value], "knn") for value, _ in ordered]

    def suggest(self, transaction: Transaction, own_account: str) -> Suggestion:
        """Best single (payee, account) pair for a transaction."""
        payees = self.rank_payees(transaction)
        payee_top = payees.top
        payee = payee_top[0] if payee_top else transaction.description
        payee_confidence = payee_top[1] if payee_top else 0.0
        payee_source = payee_top[2] if payee_top else "fallback"

        accounts = self.rank_accounts(transaction, payee, own_account)
        account_top = accounts.top
        account = account_top[0] if account_top else ""
        account_confidence = account_top[1] if account_top else 0.0

        return Suggestion(
            payee=payee,
            account=account,
            confidence=min(payee_confidence, account_confidence),
            source=payee_source,
            evidence=f"payee={payee_confidence:.2f} account={account_confidence:.2f}",
        )
