"""Readers for the raw CRA T3010 extract.

The public extract is one folder per fiscal-period-end year, each holding a set
of CSVs.  The schema is *nearly* stable across 2019-2023 but not quite:

* the 2023 files are written with a UTF-8 BOM that leaks into the first column
  name, while 2019-2022 are Latin-1 with no BOM;
* the 2023 ``Ident.csv`` drops four columns (registration date, language,
  contact details) and renames three others;
* the 2023 ``Financial Section D & Schedule 6.csv`` drops the
  ``5030 Indicator`` column;
* currency is stored as text with a dollar sign, with the minus sign *outside*
  the sign (``-$1321.00``);
* dates are stored as non-zero-padded timestamps (``2019-6-30 00:00:00``).

Everything here exists to absorb that, so that downstream modules can assume a
single tidy schema.  Parsed years are cached as parquet under
``data/interim`` — the raw CSVs total ~500 MB and re-parsing them is the
slowest step in the project by an order of magnitude.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .config import INTERIM_DIR, RAW_DIR, YEARS
from .fields import FIELD_LABELS

__all__ = [
    "read_ident",
    "read_financials",
    "read_category_table",
    "load_year",
    "load_all_years",
    "parse_money",
]

log = logging.getLogger(__name__)

_IDENT_FILE = "Ident.csv"
_FINANCIAL_FILE = "Financial Section D & Schedule 6.csv"
_CATEGORY_FILE = "# Category_Sub-Category.csv"

#: Raw Ident column name (lower-cased, punctuation-stripped) -> canonical name.
#: Covers both the 2019-2022 and the 2023 spellings.
_IDENT_RENAME = {
    "bnregistration number": "bn",
    "designation code": "designation_code",
    "category code": "category_code",
    "subcategory code": "sub_category_code",
    "legal name": "legal_name",
    "account name": "account_name",
    "registration date": "registration_date",
    "language": "language",
    "mailing address": "mailing_address",
    "address line 2": "mailing_address_2",
    "city": "city",
    "province": "province",
    "postal code": "postal_code",
    "country": "country",
    "contact phone": "contact_phone",
    "contact email": "contact_email",
    "contact url": "contact_url",
}

_FINANCIAL_RENAME = {
    "bnregistration number": "bn",
    "fiscal period end": "fiscal_period_end",
    "form id": "form_id",
    "financial indicator d or 6": "filer_tier",
    "5030 indicator c or 6": "indicator_5030",
}

_MONEY_RE = re.compile(r"[$,\s]")


def _normalise(name: str) -> str:
    """Strip BOM, stray quotes and punctuation from a raw CSV column name."""
    name = name.replace("﻿", "").strip().strip('"').strip()
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a raw extract CSV as text, tolerating the BOM/encoding drift."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(path, dtype=str, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
        frame.columns = [_normalise(c) for c in frame.columns]
        return frame
    raise UnicodeDecodeError(f"could not decode {path} as utf-8-sig or latin-1")


def parse_money(series: pd.Series) -> pd.Series:
    """Convert a T3010 currency column (``"-$1321.00"``) to float.

    Blanks are preserved as ``NaN`` rather than zero.  Deciding whether a blank
    means "nil" or "not asked" needs the filer tier and belongs in
    :mod:`charity_risk.panel`, not here.

    The result is plain ``float64``, not a nullable dtype: downstream code does
    arithmetic and boolean tests on these columns, and ``pd.NA`` raises rather
    than propagating the way ``NaN`` does.
    """
    text = series.astype("string").str.replace(_MONEY_RE, "", regex=True)
    # "-$100" becomes "-100" after stripping; "($100)" is not used by CRA but
    # is cheap to support.
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def read_ident(year: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the identification file for ``year``.

    Returns one row per BN with the canonical columns.  Columns absent in a
    given year (notably ``registration_date`` in 2023) are added as all-null so
    the schema is stable across years; :func:`charity_risk.panel.build_panel`
    back-fills them from other years.
    """
    frame = _read_csv(raw_dir / str(year) / _IDENT_FILE)
    frame = frame.rename(columns=_IDENT_RENAME)
    keep = [c for c in dict.fromkeys(_IDENT_RENAME.values()) if c in frame.columns]
    frame = frame[keep].copy()

    for column in dict.fromkeys(_IDENT_RENAME.values()):
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["bn"] = frame["bn"].str.strip()
    frame["registration_date"] = pd.to_datetime(frame["registration_date"], errors="coerce")
    for column in ("designation_code", "category_code", "sub_category_code",
                   "province", "country", "language"):
        frame[column] = frame[column].astype("string").str.strip().str.upper()

    frame["year"] = year
    # A handful of BNs appear twice with identical content; keep the first.
    return frame.drop_duplicates(subset="bn", keep="first").reset_index(drop=True)


def read_financials(year: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the Section D / Schedule 6 financial file for ``year``.

    All currency lines listed in :data:`charity_risk.fields.FIELD_LABELS` are
    parsed to float and *kept under their line code* — renaming to human labels
    happens in :mod:`charity_risk.panel` once the tier logic has been applied.
    Rows are **not** de-duplicated here; see
    :func:`charity_risk.panel.dedupe_returns`.
    """
    frame = _read_csv(raw_dir / str(year) / _FINANCIAL_FILE)
    frame = frame.rename(columns=_FINANCIAL_RENAME)

    if "indicator_5030" not in frame.columns:  # absent in 2023
        frame["indicator_5030"] = pd.NA

    frame["bn"] = frame["bn"].str.strip()
    frame["fiscal_period_end"] = pd.to_datetime(frame["fiscal_period_end"], errors="coerce")
    frame["filer_tier"] = frame["filer_tier"].astype("string").str.strip().str.upper()
    frame["accounting_basis"] = frame.get("4020", pd.Series(pd.NA, index=frame.index))
    frame["accounting_basis"] = frame["accounting_basis"].astype("string").str.strip().str.upper()

    money = {code: parse_money(frame[code]) for code in FIELD_LABELS if code in frame.columns}
    parsed = pd.DataFrame(money, index=frame.index)

    meta = frame[["bn", "fiscal_period_end", "form_id", "filer_tier",
                  "indicator_5030", "accounting_basis"]].copy()
    meta["year"] = year

    missing = sorted(set(FIELD_LABELS) - set(parsed.columns))
    if missing:
        log.warning("year %s is missing financial lines: %s", year, missing)
        for code in missing:
            parsed[code] = pd.NA

    return pd.concat([meta, parsed[list(FIELD_LABELS)]], axis=1)


def read_category_table(year: int = 2021, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the category / sub-category lookup.

    The table is duplicated identically in every year folder, so any one will
    do.  Returns one row per category code with the English description and the
    broader "charity type" grouping used as the sector control in the Trussel
    (2002) specification.
    """
    frame = _read_csv(raw_dir / str(year) / _CATEGORY_FILE)
    frame = frame.rename(columns={
        "category code": "category_code",
        "category english desc": "category_desc",
        "subcategory code": "sub_category_code",
        "subcategory english desc": "sub_category_desc",
        "charity type english desc": "charity_type",
    })
    columns = ["category_code", "category_desc", "charity_type"]
    out = frame[columns].drop_duplicates(subset="category_code", keep="first")
    out["category_code"] = out["category_code"].astype("string").str.strip().str.upper()
    return out.reset_index(drop=True)


def load_year(year: int, raw_dir: Path = RAW_DIR, cache_dir: Path | None = INTERIM_DIR,
              refresh: bool = False) -> pd.DataFrame:
    """Read and join Ident + financials for one year, with a parquet cache.

    Set ``refresh=True`` to bypass an existing cache entry, or ``cache_dir=None``
    to disable caching entirely.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"t3010_{year}.parquet"
        if cache_path.exists() and not refresh:
            return pd.read_parquet(cache_path)

    financials = read_financials(year, raw_dir=raw_dir)
    ident = read_ident(year, raw_dir=raw_dir).drop(columns=["year"])
    merged = financials.merge(ident, on="bn", how="left", validate="many_to_one")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_path, index=False)
    return merged


def load_all_years(years: tuple[int, ...] = YEARS, raw_dir: Path = RAW_DIR,
                   cache_dir: Path | None = INTERIM_DIR, refresh: bool = False) -> pd.DataFrame:
    """Stack :func:`load_year` over ``years`` into one long frame."""
    frames = [load_year(year, raw_dir=raw_dir, cache_dir=cache_dir, refresh=refresh)
              for year in years]
    return pd.concat(frames, ignore_index=True)
