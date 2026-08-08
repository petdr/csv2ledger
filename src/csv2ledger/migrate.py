"""One-time conversion of an .icsv2ledgerrc into banks.toml.

The translation is deliberately mechanical: every key is carried across with the same
meaning, including the negative-index sign-inversion convention. That is what lets the
differential test (old tool vs new tool, byte-for-byte) actually prove the migration.
"""

from __future__ import annotations

import configparser
from pathlib import Path

# Keys icsv2ledger understands that have a direct equivalent here. Anything outside this
# set is reported rather than silently dropped -- a quietly ignored key would change
# import behaviour with no warning.
_KEY_MAP = {
    "account": "account",
    "currency": "currency",
    "credit_currency": "credit_currency",
    "date": "date",
    "desc": "description",
    "credit": "credit",
    "debit": "debit",
    "csv_date_format": "csv_date_format",
    "ledger_date_format": "ledger_date_format",
    "skip_lines": "skip_lines",
    "delimiter": "delimiter",
    "encoding": "encoding",
    "csv_decimal_comma": "csv_decimal_comma",
    "ledger_decimal_comma": "ledger_decimal_comma",
    "cleared_character": "cleared_character",
    "default_expense": "default_account",
    "ledger_file": "journal",
    "template_file": "template",
    "mapping_file": "legacy_mapping_file",
}

_INT_KEYS = {"date", "credit", "debit", "skip_lines"}
_BOOL_KEYS = {"csv_decimal_comma", "ledger_decimal_comma"}
# Keys that are per-run behaviour rather than per-bank format; the CLI owns these now.
_RUNTIME_KEYS = {"quiet", "reverse", "clear_screen", "tags", "skip_older_than", "accounts_file"}


class MigrationError(Exception):
    pass


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parse_description(raw: str) -> list[int] | int:
    """icsv2ledger allows 'desc = 2, 3' meaning "join columns 2 and 3 with a space"."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise MigrationError(f"empty desc value: {raw!r}")
    indices = []
    for part in parts:
        try:
            indices.append(int(part))
        except ValueError as exc:
            raise MigrationError(f"non-numeric desc column {part!r}") from exc
    return indices[0] if len(indices) == 1 else indices


def convert(rc_path: str | Path, *, root: str | Path | None = None) -> tuple[str, list[str]]:
    """Convert an .icsv2ledgerrc to banks.toml text.

    Relative paths in the old config were resolved against its own directory, so that
    directory is recorded as ``defaults.root``. The new config can then live anywhere.

    Returns the TOML document and a list of human-readable warnings.
    """
    path = Path(rc_path).expanduser()
    if not path.is_file():
        raise MigrationError(f"config file not found: {path}")

    # Section names (the bank names) keep their case; configparser only folds keys, and
    # keys are matched case-insensitively below anyway.
    parser = configparser.RawConfigParser()
    parser.read(path, encoding="utf-8")

    warnings: list[str] = []
    banks: dict[str, dict[str, object]] = {}

    for section in parser.sections():
        if section.endswith("_addons"):
            warnings.append(
                f"[{section}]: addon columns are not supported yet; section skipped"
            )
            continue
        converted: dict[str, object] = {}
        for key, raw in parser.items(section):
            key_lower = key.strip().lower()
            if key_lower in _RUNTIME_KEYS:
                continue
            target = _KEY_MAP.get(key_lower)
            if target is None:
                warnings.append(f"[{section}]: unrecognised key {key!r} ignored")
                continue
            raw = raw.strip()
            if target == "description":
                converted[target] = _parse_description(raw)
            elif target in _INT_KEYS:
                try:
                    converted[target] = int(raw)
                except ValueError as exc:
                    raise MigrationError(f"[{section}] {key}: expected an integer, got {raw!r}") from exc
            elif target in _BOOL_KEYS:
                converted[target] = raw.lower() in {"1", "true", "yes", "on"}
            else:
                converted[target] = raw
        if "account" not in converted:
            warnings.append(f"[{section}]: no 'account' key; section skipped")
            continue
        banks[section] = converted

    if not banks:
        raise MigrationError(f"{path}: no usable bank sections found")

    resolved_root = Path(root).expanduser() if root else path.parent.resolve()
    return _render(banks, warnings, resolved_root), warnings


def _hoist_defaults(banks: dict[str, dict[str, object]]) -> dict[str, object]:
    """Pull keys that every bank agrees on up into [defaults], to cut repetition."""
    hoistable = {"currency", "journal", "template", "ledger_date_format", "default_account"}
    defaults: dict[str, object] = {}
    for key in sorted(hoistable):
        values = {repr(bank.get(key)) for bank in banks.values()}
        if len(values) == 1 and next(iter(banks.values())).get(key) is not None:
            defaults[key] = next(iter(banks.values()))[key]
            for bank in banks.values():
                bank.pop(key, None)
    return defaults


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return _toml_str(str(value))


# Written in a fixed order so the generated file is stable and diffable across re-runs.
_EMIT_ORDER = [
    "account",
    "currency",
    "credit_currency",
    "date",
    "description",
    "credit",
    "debit",
    "csv_date_format",
    "ledger_date_format",
    "skip_lines",
    "delimiter",
    "encoding",
    "csv_decimal_comma",
    "ledger_decimal_comma",
    "cleared_character",
    "default_account",
    "journal",
    "template",
    "legacy_mapping_file",
    "mappings_dir",
]


def _mappings_dir(banks: dict[str, dict[str, object]], root: Path) -> str | None:
    """The directory the per-bank mapping files live in, if they share one.

    The new tool reads every mapping file as training data, not just the current bank's,
    so it needs the directory rather than the individual paths.
    """
    parents = {
        str(Path(str(bank["legacy_mapping_file"])).parent)
        for bank in banks.values()
        if bank.get("legacy_mapping_file")
    }
    if len(parents) != 1:
        return None
    directory = Path(next(iter(parents)))
    if directory.is_absolute():
        try:
            return str(directory.relative_to(root))
        except ValueError:
            return str(directory)
    return str(directory)


def _render(banks: dict[str, dict[str, object]], warnings: list[str], root: Path) -> str:
    defaults = _hoist_defaults(banks)
    lines = [
        "# Generated by `csv2ledger migrate` from .icsv2ledgerrc.",
        "#",
        "# Column references are 1-based; 0 means absent. A negative index reads that",
        "# column and inverts the sign, matching icsv2ledger. Columns may also be named,",
        "# e.g. `date = \"Date\"`, resolved against the CSV header row.",
        "",
        "[defaults]",
        f"# Relative paths below resolve against this directory.",
        f"root = {_toml_str(str(root))}",
    ]
    mappings_dir = _mappings_dir(banks, root)
    if mappings_dir:
        defaults["mappings_dir"] = mappings_dir
    for key in _EMIT_ORDER:
        if key in defaults:
            lines.append(f"{key} = {_format_value(defaults[key])}")
    lines.extend(
        [
            "",
            "# The learning store. Every confirmation is appended here and shared by all",
            "# banks, which is what replaces copying rules between mapping files.",
            "# Defaults to ~/.local/share/csv2ledger/ when unset.",
            '# training_file = "training.jsonl"',
            '# fingerprint_file = "imported.log"',
        ]
    )
    lines.append("")

    for name in sorted(banks):
        bank = banks[name]
        lines.append(f"[banks.{name}]")
        for key in _EMIT_ORDER:
            if key in bank:
                lines.append(f"{key} = {_format_value(bank[key])}")
        for key in sorted(set(bank) - set(_EMIT_ORDER)):
            lines.append(f"{key} = {_format_value(bank[key])}")
        lines.append("")

    if warnings:
        lines.append("# Warnings raised during migration:")
        lines.extend(f"#   {warning}" for warning in warnings)
        lines.append("")

    return "\n".join(lines)
