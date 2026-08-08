"""Bank configuration: loading, validation, and column-reference semantics.

Column references keep icsv2ledger's conventions, because the migrated config has to
mean exactly what the old one meant:

  * indices are 1-based
  * index 0 means "this column is absent"
  * a negative index means "read column abs(i), then invert the sign"

A column may also be named (``date = "Date"``), resolved against the CSV header row.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TEMPLATE = """
{date} {cleared_character} {payee}
    {debit_account:<60}    {debit_currency} {debit}
    {credit_account:<60}    {credit_currency} {credit}
"""


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class ColumnRef:
    """A reference to one CSV column, by 1-based index or by header name.

    ``invert`` carries the old negative-index convention as an explicit flag so that
    downstream code never has to remember what a negative index meant.
    """

    index: int | None = None
    name: str | None = None
    invert: bool = False

    @property
    def absent(self) -> bool:
        return self.index is None and self.name is None

    @classmethod
    def parse(cls, value: object, *, what: str) -> ColumnRef:
        if value is None or value == "" or value == 0:
            return cls()
        if isinstance(value, bool):
            raise ConfigError(f"{what}: expected a column index or name, got a boolean")
        if isinstance(value, int):
            return cls(index=abs(value), invert=value < 0)
        if isinstance(value, str):
            # Tolerate numeric strings so a hand-edited "3" behaves like 3.
            stripped = value.strip()
            if stripped.lstrip("-").isdigit():
                return cls.parse(int(stripped), what=what)
            return cls(name=stripped)
        raise ConfigError(f"{what}: expected a column index or name, got {type(value).__name__}")

    def resolve(self, header: list[str] | None) -> int:
        """Return the 0-based position of this column, or -1 if absent.

        Raises ConfigError if a named column is not present in the header.
        """
        if self.absent:
            return -1
        if self.index is not None:
            return self.index - 1
        name = self.name
        assert name is not None  # guaranteed by `absent` above
        if header is None:
            raise ConfigError(
                f"column {name!r} is referenced by name but the CSV has no header row "
                f"(set skip_lines so the header is consumed, or use a numeric index)"
            )
        wanted = name.strip().casefold()
        for position, cell in enumerate(header):
            if cell.strip().casefold() == wanted:
                return position
        raise ConfigError(f"column {name!r} not found in CSV header: {header}")


@dataclass
class BankConfig:
    """Everything needed to turn one bank's CSV into ledger entries."""

    name: str
    account: str
    currency: str = "AUD"
    credit_currency: str | None = None

    date: ColumnRef = field(default_factory=lambda: ColumnRef(index=1))
    # icsv2ledger allowed several description columns joined by a space.
    description: tuple[ColumnRef, ...] = ()
    credit: ColumnRef = field(default_factory=ColumnRef)
    debit: ColumnRef = field(default_factory=ColumnRef)

    csv_date_format: str = "%d/%m/%Y"
    ledger_date_format: str = "%Y/%m/%d"

    skip_lines: int = 0
    delimiter: str = ","
    encoding: str = "utf-8"
    csv_decimal_comma: bool = False
    ledger_decimal_comma: bool = False

    cleared_character: str = "*"
    default_account: str = "Expenses:Unknown"
    journal: Path | None = None
    template_file: Path | None = None
    legacy_mapping_file: Path | None = None

    # Shared across banks: the whole point of the new store is that a merchant taught on
    # one card is immediately known to the others.
    mappings_dir: Path | None = None
    training_file: Path | None = None
    fingerprint_file: Path | None = None

    def template(self) -> str:
        if self.template_file is None:
            return DEFAULT_TEMPLATE
        return self.template_file.read_text(encoding="utf-8")

    def effective_credit_currency(self) -> str:
        return self.credit_currency if self.credit_currency is not None else self.currency


def _as_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def load(config_path: str | Path) -> dict[str, BankConfig]:
    """Load banks.toml and return every configured bank, keyed by name.

    Relative paths resolve against ``defaults.root`` when set, otherwise against the
    config file's own directory. ``root`` exists so a config migrated out of a data
    directory can live somewhere else (e.g. ~/.config) while still pointing at the
    journal and mapping files where they actually are.
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    defaults = raw.get("defaults", {})
    root = defaults.get("root")
    base = Path(root).expanduser() if root else path.parent
    if not base.is_dir():
        raise ConfigError(f"{path}: defaults.root is not a directory: {base}")
    banks_table = raw.get("banks", {})
    if not banks_table:
        raise ConfigError(f"{path}: no [banks.NAME] sections found")

    banks: dict[str, BankConfig] = {}
    for name, section in banks_table.items():
        merged = {**defaults, **section}
        banks[name] = _build_bank(name, merged, base)
    return banks


def _build_bank(name: str, merged: dict, base: Path) -> BankConfig:
    merged.pop("root", None)  # a [defaults]-only key, not a bank field
    if "account" not in merged:
        raise ConfigError(f"bank {name!r}: missing required key 'account'")

    description = merged.get("description", 2)
    if not isinstance(description, list):
        description = [description]
    description_refs = tuple(
        ColumnRef.parse(item, what=f"bank {name!r} description") for item in description
    )
    if all(ref.absent for ref in description_refs):
        raise ConfigError(f"bank {name!r}: 'description' must reference at least one column")

    credit = ColumnRef.parse(merged.get("credit", 0), what=f"bank {name!r} credit")
    debit = ColumnRef.parse(merged.get("debit", 0), what=f"bank {name!r} debit")
    if credit.absent and debit.absent:
        raise ConfigError(f"bank {name!r}: at least one of 'credit' or 'debit' must be set")

    return BankConfig(
        name=name,
        account=merged["account"],
        currency=merged.get("currency", "AUD"),
        credit_currency=merged.get("credit_currency"),
        date=ColumnRef.parse(merged.get("date", 1), what=f"bank {name!r} date"),
        description=description_refs,
        credit=credit,
        debit=debit,
        csv_date_format=merged.get("csv_date_format", "%d/%m/%Y"),
        ledger_date_format=merged.get("ledger_date_format", "%Y/%m/%d"),
        skip_lines=int(merged.get("skip_lines", 0)),
        delimiter=merged.get("delimiter", ","),
        encoding=merged.get("encoding", "utf-8"),
        csv_decimal_comma=bool(merged.get("csv_decimal_comma", False)),
        ledger_decimal_comma=bool(merged.get("ledger_decimal_comma", False)),
        cleared_character=merged.get("cleared_character", "*"),
        default_account=merged.get("default_account", "Expenses:Unknown"),
        journal=_as_path(base, merged.get("journal")),
        template_file=_as_path(base, merged.get("template")),
        legacy_mapping_file=_as_path(base, merged.get("legacy_mapping_file")),
        mappings_dir=_as_path(base, merged.get("mappings_dir")),
        training_file=_as_path(base, merged.get("training_file")),
        fingerprint_file=_as_path(base, merged.get("fingerprint_file")),
    )
