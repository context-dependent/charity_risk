"""Assembly of the charity-year panel.

Three things happen here, in order:

1. **De-duplication.**  About 0.2% of BNs file two returns with a fiscal period
   ending in the same calendar year, almost always because they changed their
   year-end and filed a stub period.  We keep the return with the *latest*
   fiscal period end so that stock variables (assets, liabilities) line up with
   the year label, and flag the row via ``n_returns_in_year`` so that flow
   variables from a stub period can be excluded or controlled for.

2. **Tier-aware blank handling.**  A blank currency cell is read as nil for a
   filer that was asked the question and as missing for one that was not.  See
   :mod:`charity_risk.fields` for why this distinction is not cosmetic.

3. **Presence bookkeeping.**  Whether a BN filed in each year of the window,
   which is the raw material for the exit label in
   :mod:`charity_risk.outcomes`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PANEL_END, PANEL_START, PROCESSED_DIR, YEARS
from .fields import (
    DESIGNATION_LABELS,
    FIELD_LABELS,
    REVENUE_SOURCES_CORE,
    SCHEDULE_6_ONLY,
)
from .ingest import load_all_years, read_category_table

__all__ = [
    "dedupe_returns",
    "apply_tier_missingness",
    "add_revenue_buckets",
    "presence_matrix",
    "build_panel",
    "load_panel",
    "panel_exists",
    "panel_window",
]

_PANEL_PATH = PROCESSED_DIR / "panel.parquet"


def dedupe_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce to one return per ``(bn, year)``.

    Preference order: latest fiscal period end, then Schedule 6 over Section D
    (more information), then largest total revenue.  Adds ``n_returns_in_year``.
    """
    counts = frame.groupby(["bn", "year"], observed=True)["fiscal_period_end"].transform("size")
    frame = frame.assign(n_returns_in_year=counts.astype("int16"))

    order = frame.assign(
        _tier_rank=frame["filer_tier"].eq("6").fillna(False).astype("int8"),
        _revenue=frame["4700"].fillna(-np.inf),
    ).sort_values(
        ["bn", "year", "fiscal_period_end", "_tier_rank", "_revenue"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    deduped = order.drop_duplicates(subset=["bn", "year"], keep="last")
    return deduped.drop(columns=["_tier_rank", "_revenue"]).reset_index(drop=True)


def apply_tier_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill blanks that mean nil; leave blanks that mean "never asked".

    The rule, stated plainly, is that a blank means two different things
    depending on whether the question was asked at all:

    * **asked and left blank = nil.**  A Schedule 6 filer that reports total
      revenue but leaves line 4540 empty received no federal government revenue.
      That is a reported zero, not an unknown, and it is filled with ``0.0``.
    * **never asked = missing.**  A Section D filer has no line 4540 on its form.
      Its federal government revenue is genuinely unknown — whatever it was, it
      is bundled into the totals — and it stays ``NaN``.

    Also emits ``share_nil_lines``: of the lines the filer's tier was asked, the
    fraction reported as nil.  Because a blank *is* a zero, this is not a
    data-quality measure.  It counts how many kinds of revenue and expenditure
    the organisation reports at all, and so proxies the breadth of its
    operations: a charity reporting nil on nearly every line is a simple or
    dormant one, not a careless filer.  Expressed as a share rather than a count
    because the two tiers are asked different numbers of questions.
    """
    frame = frame.copy()
    codes = [code for code in FIELD_LABELS if code in frame.columns]
    is_section_d = frame["filer_tier"].eq("D").fillna(False).to_numpy()
    all_collected = np.ones(len(frame), dtype=bool)

    # Column at a time: the whole-frame version needs four simultaneous copies
    # of an 85-column float matrix, which is the peak-memory step of the build.
    n_nil = np.zeros(len(frame), dtype="int32")
    n_collected = np.zeros(len(frame), dtype="int32")
    for code in codes:
        # `np.array` rather than `.to_numpy()`: under copy-on-write the latter
        # can hand back a read-only view of the column's own buffer.
        values = np.array(pd.to_numeric(frame[code], errors="coerce"), dtype="float64")
        blank = np.isnan(values)
        collected = ~is_section_d if code in SCHEDULE_6_ONLY else all_collected
        values[blank & collected] = 0.0
        values[~collected] = np.nan
        frame[code] = values
        n_nil += (blank & collected)
        n_collected += collected

    frame["share_nil_lines"] = (n_nil / np.maximum(n_collected, 1)).astype("float64")
    return frame


def add_revenue_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the harmonised revenue decomposition used for concentration indices.

    Section D and Schedule 6 filers are mapped onto the same eight buckets
    (:data:`charity_risk.fields.REVENUE_SOURCES_CORE`), so that a Tuckman-Chang
    concentration index is comparable across tiers.  Column names are prefixed
    ``rev_``.
    """
    frame = frame.copy()
    is_section_d = frame["filer_tier"].eq("D").fillna(False)

    for bucket, by_tier in REVENUE_SOURCES_CORE.items():
        d_total = sum(frame[code].fillna(0.0) for code in by_tier["D"] if code in frame.columns)
        six_total = sum(frame[code].fillna(0.0) for code in by_tier["6"] if code in frame.columns)
        frame[f"rev_{bucket}"] = np.where(is_section_d, d_total, six_total)

    bucket_cols = [f"rev_{bucket}" for bucket in REVENUE_SOURCES_CORE]
    frame["rev_bucket_total"] = frame[bucket_cols].sum(axis=1)
    return frame


def presence_matrix(frame: pd.DataFrame, years: tuple[int, ...] = YEARS) -> pd.DataFrame:
    """Return a BN x year 0/1 indicator of whether a return was filed."""
    flags = (
        frame.assign(filed=1)
        .pivot_table(index="bn", columns="year", values="filed", aggfunc="max", fill_value=0)
        .reindex(columns=list(years), fill_value=0)
        .astype("int8")
    )
    flags.columns = [f"filed_{year}" for year in years]
    return flags


def _add_presence_columns(frame: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    flags = presence_matrix(frame, years=years)
    filed_cols = list(flags.columns)
    year_index = np.array(years)

    matrix = flags[filed_cols].to_numpy()
    first = np.where(matrix.any(axis=1), year_index[matrix.argmax(axis=1)], -1)
    last = np.where(matrix.any(axis=1),
                    year_index[matrix.shape[1] - 1 - matrix[:, ::-1].argmax(axis=1)], -1)
    n_filed = matrix.sum(axis=1)
    # A "gap" is a non-filing year strictly inside the observed span.
    span = (last - first + 1)
    flags = flags.assign(
        first_filing_year=first,
        last_filing_year=last,
        n_years_filed=n_filed,
        has_filing_gap=(span > n_filed),
    )
    return frame.merge(flags.reset_index(), on="bn", how="left", validate="many_to_one")


def build_panel(years: tuple[int, ...] = YEARS, refresh: bool = False,
                save: bool = True) -> pd.DataFrame:
    """Build the full charity-year panel from the raw extract.

    The result is one row per ``(bn, year)`` with:

    * identification and sector attributes (designation, category, province,
      registration date, organisation age);
    * every T3010 financial line, tier-aware zero-filled, under its human label
      *and* under its raw line code (the raw code is kept because the
      literature refers to lines by number);
    * the harmonised ``rev_*`` revenue buckets;
    * filing-presence bookkeeping (``filed_2019`` ... ``last_filing_year``).

    Set ``save=False`` to skip writing ``data/processed/panel.parquet``.
    """
    raw = load_all_years(years=years, refresh=refresh)
    panel = dedupe_returns(raw)
    del raw
    panel = apply_tier_missingness(panel)
    panel = add_revenue_buckets(panel)

    # Registration date is absent from the 2023 Ident file; it is time-invariant,
    # so carry it (and the other stable attributes) across years within a BN.
    stable = ["registration_date", "designation_code", "category_code",
              "sub_category_code", "province", "country", "language", "legal_name"]
    panel = panel.sort_values(["bn", "year"], kind="mergesort")
    for column in stable:
        panel[column] = panel.groupby("bn", observed=True)[column].ffill().bfill()

    panel["designation"] = (
        panel["designation_code"].map(DESIGNATION_LABELS).fillna("unknown").astype("category")
    )
    categories = read_category_table()
    panel = panel.merge(categories, on="category_code", how="left")
    panel["category_desc"] = panel["category_desc"].fillna("unknown").astype("category")
    panel["charity_type"] = panel["charity_type"].fillna("unknown").astype("category")
    panel["province"] = panel["province"].fillna("unknown").astype("category")

    panel["age_years"] = (
        panel["year"] - panel["registration_date"].dt.year
    ).astype("float64").clip(lower=0)
    panel["fiscal_month"] = panel["fiscal_period_end"].dt.month.astype("float64")
    panel["is_section_d"] = panel["filer_tier"].eq("D").fillna(False).astype("int8")
    panel["is_accrual"] = panel["accounting_basis"].eq("A").fillna(False).astype("int8")

    panel = _add_presence_columns(panel, years=years)

    # Convenience aliases used throughout the feature code.
    panel["total_assets"] = panel["4200"]
    panel["total_liabilities"] = panel["4350"]
    panel["net_assets"] = panel["4200"] - panel["4350"]
    panel["total_revenue"] = panel["4700"]
    panel["total_expenditures"] = panel["5100"]
    panel["expenditures_before_donee_gifts"] = panel["4950"]

    panel = panel.sort_values(["bn", "year"], kind="mergesort").reset_index(drop=True)
    if save:
        _PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(_PANEL_PATH, index=False)
    return panel


def load_panel(years: list[int] | tuple[int, ...] | None = None,
               columns: list[str] | tuple[str, ...] | None = None,
               refresh: bool = False) -> pd.DataFrame:
    """Load the cached panel, building it on first use.

    ``years`` and ``columns`` are pushed down to the parquet reader.  The full
    panel is ~120 columns wide across five years; projecting it is the
    difference between a 60 MB read and a 600 MB one.
    """
    if not _PANEL_PATH.exists() or refresh:
        build_panel(refresh=refresh)
    read_columns = None if columns is None else list(dict.fromkeys(["bn", "year", *columns]))
    filters = None if years is None else [("year", "in", list(years))]
    return pd.read_parquet(_PANEL_PATH, columns=read_columns, filters=filters)


def panel_exists() -> bool:
    """Whether the cached panel has been written."""
    return _PANEL_PATH.exists()


def panel_window() -> tuple[int, int]:
    """Return the inclusive ``(first, last)`` year of the observation window."""
    return PANEL_START, PANEL_END
