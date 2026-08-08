# How the classification works

There is no trained classifier in the usual sense — no weights, no gradient descent, no
fit/predict cycle. This is **instance-based learning** (also called memory-based or lazy
learning): the model *is* the corpus of labelled examples, and "training" means adding rows
to it. That choice is forced by the shape of the data, which the last section comes back to.

## The pipeline

```
raw description
  → normalize()        strip per-transaction noise      normalize.py:63
  → FeatureSpace       hash into a sparse vector        features.py:120
  → cosine vs matrix   find the k nearest labelled rows suggest.py:290
  → weighted vote      similarity-weighted tally        suggest.py:308
```

### Normalisation

Most of the work, and not ML at all. `normalize.py:21` is an ordered list of 18 targeted
substitutions — card numbers before digit runs, dates before bare numbers — that turn
per-transaction noise into `#`. So

```
ALDI STORES WES T MELBOURNVI - EFTPOS Purchase with Cash Out
  - Receipt 98988 - Cash amount $50.00<BR/>Date 23 Jul 2026 Card 462263xxxxxx2739
→ ALDI STORES WES T MELBOURNVI EFTPOS PURCHASE WITH CASH OUT RECEIPT CASHOUT CARD
```

Every Aldi cash-out now collapses to one key. That single step is why ~6,300 hand-written
rules were needed before and are not now.

The risk runs the other way too — strip too much and distinct merchants merge — so each rule
is aimed at a specific observed pattern rather than being a blanket digit purge, and they are
pinned by tests against real descriptions.

### Vectorisation

`features.py` turns the normalised key into a point in a ~400k-dimensional sparse space,
built from four concatenated blocks:

| Block | What it buys |
|---|---|
| char_wb 3–5 grams over the whole string | the generalisation — `WW METRO # MELBOURNE VI` sits close to `WW METRO # CARLTON VI` |
| char_wb 3–5 grams over the first 24 chars, ×1.5 | card descriptions put the merchant first and the suburb second; without this the shared `MELBOURNE VI` tail pulls unrelated merchants together |
| word 1–2 grams | whole-token matches |
| bank + debit/credit, ×0.15 | a tie-breaker, deliberately too weak to outvote the description |

## The one thing that is actually fitted

IDF weights, at `features.py:100`. That is all — a vector of 393,472 floats, one per feature
column, ranging 1.00 to 9.73, which `transform()` multiplies each row by. No classes, no
decision boundary.

IDF measures how rare each feature is across the corpus and scales it up accordingly. Real
numbers from the 6,167 examples:

| token | appears in | idf weight |
|---|---|---|
| `#` | 1,571 docs | 2.37 |
| `RECEIPT` | 1,151 | 2.68 |
| `MELBOURNE` | 986 | 2.83 |
| `PURCHASE` | 481 | 3.55 |
| `WOOLWORTHS` | 14 | 7.02 |
| `KATHMANDU` | 7 | 7.65 |
| `ZOUKI` | 1 | 9.03 |

The formula is `ln((1+n)/(1+df)) + 1`. `MELBOURNE` still counts, but `KATHMANDU` counts nearly
three times as much per occurrence — and that ratio is measured from the data, not
hand-specified. It is what makes cosine similarity track the merchant name instead of the
shared formatting.

A `HashingVectorizer` is used rather than `TfidfVectorizer` specifically to avoid a *second*
fitted thing. A hashing vectoriser has no vocabulary: a token's column is
`hash(token) % n_features`, computed rather than looked up, so a word never seen before still
lands somewhere deterministic. With a fitted vocabulary, adding an example containing a new
token would change the feature space and invalidate every vector already in the index. This is
the design decision the whole incremental story rests on.

### The scope of the freeze

Within a session the IDF vector is fixed: it is fitted in `Suggester.__init__` and never
recomputed, so a row appended by `add()` is weighted with the same vector as every row already
in the index. That coherence is the point — a query weighted differently from the rows it is
compared against is not measuring anything meaningful.

Across runs it is not frozen at all. `_build_suggester` folds the whole training store into the
corpus (`cli.py:161`) *before* constructing the Suggester, so every import refits IDF over
mappings plus everything confirmed to date. There is no long-term staleness, and no retraining
step to run.

### Where the freeze actually costs something

Of the 393,472 columns, **306,311 were never touched during fit** — 78% of the space. All of
them carry the maximum weight, 9.73, because `df = 0` maximises the formula. Any character
n-gram absent from the corpus at fit time is therefore treated as maximally informative.

For a new merchant that is correct: `ZZQWERTYNEWMERCHANT` hashes into 54 buckets averaging 7.79
weight, so a novel name dominates the similarity computation, which is exactly right because
the merchant name is the signal.

The failure case is new **boilerplate**. If a bank changes its export format and starts
prefixing every row with a phrase that was not in the corpus, that phrase carries near-maximum
weight on every transaction in the file. Instead of being discounted as noise it becomes the
strongest shared feature between unrelated rows, pulling everything toward everything.

The blast radius is one import. The next run refits with those rows included, IDF sees the
phrase on hundreds of documents and discounts it, and the problem disappears without
intervention. So the trigger to watch for is not elapsed time or corpus size but **a bank
changing its description format**, and the symptom is one unusually bad import followed by
normal ones. `csv2ledger retrain` reports tokens matching that signature — common in recent
confirmations, absent from the seed corpus — so you can tell that is what happened.

## Where the training data comes from

Two sources at startup (`seed.py:157`), plus one that grows:

- **`mappings/*`** — the only place a raw bank description sits next to its label, so the only
  source that can train description→payee. 14,562 rows deduped to 6,341, because the per-bank
  files share huge blocks (COMMBANK ∩ ONE = 3,465 identical lines) and without deduping a rule
  would be weighted by how many banks copied it rather than by how often it occurs. Literal
  rules get weight 1.0; the 1,299 regexes cannot be vectorised as-is, so `from_regex()` strips
  the metacharacters to leave the literal fragment (`/.*KATHMANDU PTY LIMITE/` →
  `KATHMANDU PTY LIMITE`) at weight 0.4 — real signal, but a pattern rather than an observed
  transaction.
- **`ross-family.dat`** — 33,875 transactions with no raw descriptions, so it cannot train the
  description model at all. What it gives is `payee_accounts`: a Counter of which accounts each
  payee has actually been posted to. That is the account-stage prior, and it covers 3,671
  payees.
- **`training.jsonl`** — everything confirmed since, at weight 1.0, or at weight `count` for
  rows that `csv2ledger retrain` has folded together (see below).

`Expenses:Unknown` is excluded from training (`seed.py:30`): it is a placeholder for
"undecided", and learning to propose it is worse than proposing nothing.

## How a suggestion is assembled

Five layers, first hit wins for the payee (`suggest.py:205`):

1. **Exact** match on the normalised key. Not simply "return the label" — `_vote()` at
   `suggest.py:48` tallies *all* examples for that key by weight. The history is not
   self-consistent (the same description filed under `Woolies` and `Woolworths` over a decade),
   so it returns the most frequent spelling, and the confidence is the winner's *vote share*. A
   genuinely split description reports 0.6 instead of claiming 1.0 and being wrong 40% of the
   time.
2. **Regex rule** from the mapping files → 0.99. Explicit decisions stay authoritative.
3. **kNN** — cosine similarity of the query against every row (a single sparse dot product,
   since everything is L2-normalised, so the cosine *is* the dot product), take the top 8 above
   0.30, tally `similarity × weight`.
4. **Literal payee name** read off the front of the description, 0.70–0.90.
5. **Journal prior**, account stage only.

One detail in the kNN worth knowing: candidates are *ordered* by summed vote, but the score
*reported* is the best single neighbour's similarity (`suggest.py:329`). The sum grows with how
many neighbours happen to agree, which would inflate confidence for common merchants regardless
of how good the match actually is. Those two orderings disagree — which is why `_merge()` must
not re-sort the list. An early version did, and auto-precision fell from 94.7% to 75.0%.

Prediction is then **chained**: description → payee, then description + *confirmed* payee →
account. Once the payee is settled the account is nearly deterministic, and correcting a wrong
payee re-ranks the account for free.

## The learning loop

This is the part that improves over time. When a suggestion is confirmed:

```python
def add(self, example: Example) -> None:        # suggest.py:189
    self.examples.append(example)
    self._exact[example.description].append(example)
    self._pending.append(self.space.transform_one(...))
    self.corpus.known_payees.add(example.payee)
```

Four effects, all O(1):

- the exact-match dict gains a key, so an identical description is now answered at 1.0
- one row is vectorised and queued; `matrix` (`suggest.py:180`) vstacks pending rows lazily on
  the next query, so a *similar* description is now within kNN reach
- the payee joins the known vocabulary
- `store.py` appends one fsynced JSON line, recording whether the suggestion was accepted or
  overridden — overrides are the signal worth having

Nothing is refitted. The next transaction in the same file already benefits. Concretely:
import a file containing `ALDI STORES 1234 CARLTON`, and by the time you reach
`ALDI STORES 5678 FITZROY` twelve rows later it is already answered — that is
`tests/test_cli.py:79`.

Repetition is evidence, and the store keeps it: confirming the same merchant fifty times means
fifty votes, which is what makes a settled decision hard to dislodge by accident.
`csv2ledger retrain` compacts those fifty rows into one carrying `count: 50`, and
`to_example()` passes the count through as the example weight, so the file shrinks while the
vote is preserved exactly. Compaction that merely dropped the repeats would turn a 50-to-1
preference into a 50/50 tie and reverse it — `tests/test_store.py` pins the invariant.

## Two honest limitations

**The literal index is built once, at `suggest.py:133`, and `add()` does not update it.** A
payee invented mid-session cannot be read off the front of a description until the next run,
when `cli.py` folds the stored confirmations back into the corpus. The exact and kNN layers
cover it within a session, so the effect is small, but it is a gap rather than a design
intention.

**One import can run on pre-format-change IDF weights**, as described above. Self-correcting,
but worth recognising when it happens.

## Why not a classifier

The obvious alternative is logistic regression or a linear SVM over the same features. The
blocker is the class distribution: **3,850 payees over 6,167 examples**. Most payees have one
or two examples. A linear model fits a weight vector per class and there is nothing to fit from
a single example — it would learn the head merchants well and be useless on the long tail,
which is precisely the tail that otherwise gets typed by hand. Nearest-neighbour does not care
how many examples a class has; with one it degrades to "the closest thing you have ever
labelled", which is the right failure mode. It also gives incrementality for free, where a
classifier wants a retrain.

The cost is inference: every query scores against all ~6,000 rows. At this corpus size that is
about a millisecond and irrelevant; at 10× it would want an approximate index.
