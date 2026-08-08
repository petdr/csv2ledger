# csv2ledger

A replacement for `icsv2ledger.py` that suggests the payee and the expense account for every
row of a bank CSV, learns from each correction, and writes ledger-cli entries.

The goal is a run of Enter presses. `icsv2ledger` matches a description against a table of
literal strings and regexes; anything unseen has no suggestion at all, which is why the
`mappings/*` files grew to 14,563 lines with 56% of them duplicated across banks. This tool
generalises across store numbers, suburbs and banks, so a merchant taught once is known
everywhere.

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run csv2ledger --help
```

## Getting started

Convert the existing rc file once:

```
uv run csv2ledger migrate ~/Dropbox/financial/.icsv2ledgerrc -o ~/.config/csv2ledger/banks.toml
```

That reads all 22 sections, hoists the shared settings into `[defaults]`, and keeps the
1-based column indices — including the `debit = -4` convention where a negative index means
"invert the sign". Then:

```
uv run csv2ledger import -a COMMBANK COMMBANK.csv -o /tmp/new.ledger
```

## How a transaction is answered

Five layers, highest confidence wins:

| Layer | Confidence | What it catches |
|---|---|---|
| Exact match on the normalised description | 1.0 | Anything imported before |
| Regex rule from `mappings/*` | 0.99 | The 1,299 patterns already written by hand |
| kNN cosine over hashed character n-grams | similarity | New store numbers, new suburbs, other banks |
| Literal payee name at the front of the description | 0.70–0.90 | Merchants known only from the journal |
| Payee→account prior from the journal | — | The account, once the payee is settled |

Two chained predictions: description → payee, then description + *confirmed* payee →
account. Correcting the payee re-ranks the account for free.

Normalisation is what makes the exact layer work at all. `ALDI STORES WES T MELBOURNVI -
EFTPOS Purchase with Cash Out - Receipt 98988 ...` collapses to a key shared by every Aldi
cash-out, because receipt numbers, card numbers, embedded dates and bare digit runs all
become `#`.

The vectoriser is a `HashingVectorizer`, deliberately — it has no fitted vocabulary, so a
confirmation appends a row to the matrix and is usable by the very next transaction in the
same session. No refit, no nightly retrain.

[CLASSIFICATION.md](CLASSIFICATION.md) explains the whole thing in detail: the feature
blocks, what is and is not fitted, how a confirmation feeds back, and why this is
nearest-neighbour rather than a trained classifier.

## Prompting

```
  2026/02/01   -15.00   ALDI STORES 1234 CARLTON VIC
Payee   > Aldi
Account > Expenses:Food:Groceries
```

Enter accepts. Tab completes over known payees and accounts. `:s` skips the transaction,
`:q` stops the run keeping everything already classified, `:?` prints help, and a bare
number picks that entry from the ranked list.

`--auto 0.95` commits anything at or above that confidence without asking, provided the
margin to the runner-up is at least 0.10. A duplicate warning always forces a prompt.

## Learning store

Every confirmation appends one JSON line to `training.jsonl` (fsynced), recording whether
the suggestion was accepted or overridden. Overrides are the valuable signal. `mappings/*`
are read — as rules and as training data — but never written to, which is what stops the
duplication from growing.

`csv2ledger stats` summarises what has been learned.

## Maintenance: `csv2ledger retrain`

```
uv run csv2ledger retrain
```

It compacts the training store, rebuilds the index and reports what it found.

**What it does not do, despite the name.** It does not refit the model, because there is
nothing to refit — every `import` already rebuilds the index and refits the IDF weights over
the full corpus at startup (`cli.py:170`). There is no stale-weights problem to solve, and
running `retrain` will never change a suggestion.

So it is housekeeping, and it is worth running in three situations:

- **The store has got large.** It is append-only, so confirming the same merchant 200 times
  writes 200 rows. Compaction folds repeats of an identical decision into a single row with a
  `count`, which is passed through as the example's vote weight — so the file shrinks and
  every suggestion stays bit-for-bit the same. Startup time scales with the row count, so
  this is the reason that will actually bite you.
- **After a bank changes its CSV format.** `retrain` reports tokens that are now common in
  your confirmations but were absent from the mapping corpus, which is the signature of new
  boilerplate rather than a new merchant. See [CLASSIFICATION.md](CLASSIFICATION.md) for why
  that matters: the first import after such a change runs with weights that treat the new
  boilerplate as maximally informative. The report is retrospective — the rebuild has already
  absorbed it — but it tells you which import to be suspicious of.
- **To check the index still builds** after editing `banks.toml` or the mapping files, without
  running an import.

Use `--no-compact` to get the rebuild and the drift report while leaving the store untouched.

One thing `retrain` deliberately will not do is help you change your mind about a payee
spelling. If a description has been confirmed as `Woolies` fifty times, one correction to
`Woolworths` is outvoted fifty to one, and compaction preserves that ratio rather than
flattening it — flattening would silently reverse settled decisions. Editing
`training.jsonl` directly is the honest way to make that change; it is line-oriented JSON
precisely so that it stays greppable and hand-editable.

## Duplicate protection

The wrapper scripts download overlapping date ranges, so the same row arrives in more than
one CSV. Two checks:

- a fingerprint log of everything committed, which makes a straight re-import a no-op
- a scan of the target journal for a transaction on the same date for the same amount,
  which can only warn, because the journal stores the payee rather than the bank description

Both count occurrences rather than storing a set: two identical coffees on one day are two
real transactions.

## Measured results

**Differential test** (`scripts/difftest.sh`) — `--rules-only` output is byte-identical to
`icsv2ledger.py` across four column layouts, two date formats and both sign conventions:

```
COMMBANK   OK   (530 lines identical)
MSAV       OK   (665 lines identical)
EVERYDAY   OK   (28535 lines identical)
MAX        OK   (6175 lines identical)
```

**Held-out backtest** (`scripts/eval.py`, random 80/20 split of the 6,167-example seed
corpus):

| | |
|---|---|
| payee top-1 | 53.0% |
| payee top-3 | 61.7% |
| account top-1, payee given | 83.0% |
| both correct (pure Enter) | 43.2% |
| no suggestion at all | 5.2% |

The split is harsh: 44% of held-out payees never appear in training at all, so the ceiling
is 55.7% and the model reaches 64.8% of what is reachable.

**Cold-start on real files** (`scripts/realcheck.py` — every description treated as never
seen, the bank's own rule layer disabled, the exact row hidden):

| bank | payee | account | both |
|---|---|---|---|
| COMMBANK (106) | 70.8% | 86.8% | 67.9% |
| EVERYDAY (5,699) | 50.1% | 83.5% | 48.8% |
| MSAV (133) | 60.9% | 88.0% | 60.2% |
| MAX (1,233) | 55.2% | 98.7% | 55.1% |

**Day-to-day** (`--auto 0.95 --dry-run`, descriptions already known, which is the actual
experience): COMMBANK 98/106, MSAV 132/133 and MAX 1,220/1,235 rows commit without
attention — 92–99% of a file.

## Tests

```
uv run pytest          # 85 tests
bash scripts/difftest.sh
```
