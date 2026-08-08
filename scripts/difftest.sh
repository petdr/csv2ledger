#!/usr/bin/env bash
# Differential test: prove csv2ledger reproduces icsv2ledger byte-for-byte.
#
# Both tools run against an isolated copy of the data directory, because
# icsv2ledger appends to its mapping files as it runs and must not be pointed at
# the real ones. The reference copy of icsv2ledger.py is patched to stub out its
# `ledger` subprocess call, which only populates tab-completion candidates.
#
# Usage: scripts/difftest.sh [BANK ...]      (default: COMMBANK MSAV EVERYDAY MAX)

set -uo pipefail

SRC="${CSV2LEDGER_DATA:-$HOME/Dropbox/financial}"
WORK="$(mktemp -d /tmp/csv2ledger-difftest.XXXXXX)"
PYTHON="${REFERENCE_PYTHON:-python3}"
BANKS=("$@")
[ ${#BANKS[@]} -eq 0 ] && BANKS=(COMMBANK MSAV EVERYDAY MAX)

cleanup() { [ -n "${KEEP_WORKDIR:-}" ] || rm -rf "$WORK"; }
trap cleanup EXIT

echo "work dir: $WORK"
mkdir -p "$WORK/mappings"
cp "$SRC"/*.csv "$SRC/TEMPLATE" "$SRC/.icsv2ledgerrc" "$WORK/" 2>/dev/null
cp "$SRC"/mappings/* "$WORK/mappings/" 2>/dev/null

# Reference tool: stub from_ledger so it does not shell out to a `ledger` binary.
sed 's/^def from_ledger(ledger_file, command):/def from_ledger(ledger_file, command):\n    return set()\n\ndef _unused_from_ledger(ledger_file, command):/' \
    "$SRC/icsv2ledger.py" > "$WORK/icsv2ledger_ref.py"

# The journal is only read for completion candidates, which are stubbed out; an
# empty file avoids copying 3.9MB per run.
: > "$WORK/ross-family.dat"

( cd "$WORK" && uv run --project "$OLDPWD" csv2ledger migrate .icsv2ledgerrc --root "$WORK" -o "$WORK/banks.toml" ) >/dev/null 2>&1 \
  || { echo "FAIL: migration errored"; exit 1; }

status=0
for bank in "${BANKS[@]}"; do
    printf '%-10s ' "$bank"
    if [ ! -f "$WORK/$bank.csv" ]; then
        echo "SKIP (no $bank.csv)"
        continue
    fi

    # Fresh mappings per bank: the reference tool mutates them as it runs.
    rm -rf "$WORK/mappings"; mkdir -p "$WORK/mappings"
    cp "$SRC"/mappings/* "$WORK/mappings/" 2>/dev/null

    # Empty lines accept every default, so the run is deterministic and unattended.
    ( cd "$WORK" && yes '' | timeout 600 "$PYTHON" icsv2ledger_ref.py -q -a "$bank" \
        "$bank.csv" "old_$bank.ledger" ) >"$WORK/$bank.reflog" 2>&1
    if [ ! -s "$WORK/old_$bank.ledger" ]; then
        echo "FAIL (reference produced no output; see $WORK/$bank.reflog)"
        status=1
        continue
    fi

    # Restore pristine mappings so csv2ledger sees the same inputs the reference started from.
    rm -rf "$WORK/mappings"; mkdir -p "$WORK/mappings"
    cp "$SRC"/mappings/* "$WORK/mappings/" 2>/dev/null

    uv run csv2ledger import -c "$WORK/banks.toml" -a "$bank" "$WORK/$bank.csv" \
        --rules-only -o "$WORK/new_$bank.ledger" 2>"$WORK/$bank.newlog"
    if [ ! -s "$WORK/new_$bank.ledger" ]; then
        echo "FAIL (csv2ledger produced no output; see $WORK/$bank.newlog)"
        status=1
        continue
    fi

    if diff -q "$WORK/old_$bank.ledger" "$WORK/new_$bank.ledger" >/dev/null; then
        echo "OK   ($(wc -l < "$WORK/new_$bank.ledger") lines identical)"
    else
        echo "DIFF ($(diff "$WORK/old_$bank.ledger" "$WORK/new_$bank.ledger" | grep -c '^[<>]') differing lines)"
        diff "$WORK/old_$bank.ledger" "$WORK/new_$bank.ledger" | head -20
        status=1
    fi
done

[ $status -eq 0 ] && echo "all banks identical"
exit $status
