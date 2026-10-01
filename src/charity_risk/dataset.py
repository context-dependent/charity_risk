"""The modelling dataset: panel + outcomes + features, with a cache.

This is the single entry point analysis code should use.  ``load_dataset()``
returns one row per charity-year with everything a model needs attached, and
the split helpers encode the temporal design described in
:mod:`charity_risk.config`.

The analysis frame is deliberately *narrower* than the panel.  The panel keeps
every T3010 line under both its code and its label, which is what you want when
auditing a ratio but not what you want in memory during a fit.  The analysis
frame keeps identifiers, outcomes, features and a handful of headline dollar
amounts, and stores ratio features as ``float32``.  Both ``load_dataset`` and
``load_panel`` accept column and year projections so that nothing larger than
the question needs to be read.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, SCORE_YEAR, TEST_YEAR, TRAIN_YEAR, YEARS
from .features import CATEGORICAL, FEATURE_GROUPS, feature_columns
from .outcomes import add_outcomes
from .panel import build_panel, load_panel, panel_exists as _PANEL_EXISTS

__all__ = [
    "ID_COLUMNS",
    "OUTCOME_COLUMNS",
    "HEADLINE_FINANCIALS",
    "SCHEDULE_6",
    "SECTION_D",
    "analysis_columns",
    "build_dataset",
    "load_dataset",
    "exit_split",
    "vulnerability_split",
    "schedule_6_split",
    "scoring_frame",
]

log = logging.getLogger(__name__)

_DATASET_PATH = PROCESSED_DIR / "analysis.parquet"

ID_COLUMNS: tuple[str, ...] = (
    "bn", "year", "legal_name", "fiscal_period_end", "filer_tier", "form_id",
    "designation", "charity_type", "category_desc", "province",
    "registration_date", "age_years", "first_filing_year",
)

#: Filing tiers, as they appear in ``filer_tier``.
SCHEDULE_6 = "6"
SECTION_D = "D"

OUTCOME_COLUMNS: tuple[str, ...] = (
    "exit_next_year", "exit_provisional", "years_observed_after",
    "vulnerable", "net_asset_change_3y", "last_filing_year", "survives_window",
)

HEADLINE_FINANCIALS: tuple[str, ...] = (
    "total_assets", "total_liabilities", "net_assets", "total_revenue",
    "total_expenditures", "expenditures_before_donee_gifts",
)


def analysis_columns() -> list[str]:
    """Every column stored in ``data/processed/analysis.parquet``."""
    features = feature_columns(list(FEATURE_GROUPS))
    ordered = list(ID_COLUMNS) + list(OUTCOME_COLUMNS) + list(HEADLINE_FINANCIALS)
    for column in features:
        if column not in ordered:
            ordered.append(column)
    return ordered


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Store ratio features as ``float32``; leave dollar amounts at full width.

    A ratio needs three significant figures, not sixteen.  Dollar lines are
    left alone because the largest charities report revenues above 10^10, where
    ``float32`` starts rounding to the nearest thousand dollars.
    """
    protected = set(HEADLINE_FINANCIALS) | {"net_asset_change_3y", "age_years"}
    for column in frame.columns:
        if column in protected:
            continue
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].astype("float32")
    return frame


def build_dataset(refresh: bool = False, save: bool = True) -> pd.DataFrame:
    """Build panel, outcomes and features end to end, then project and cache.

    Three things here are about peak memory rather than logic, and all three
    matter on a machine that will not hold several copies of a 420,000 x 90
    float64 frame:

    * the panel is written to disk and re-read as a projection, rather than
      passed through in memory;
    * :func:`charity_risk.features.build_features` copies once rather than once
      per builder, and releases the raw line codes as soon as the ratios that
      need them are computed;
    * features are built *before* outcomes, so the outcome step — which copies
      the frame — runs after the line codes have been released.

    Note that the line codes are deliberately *not* downcast to ``float32``
    here.  It would save another 80 MB, but ratios computed from narrowed
    inputs shift in the seventh significant figure, which is enough to reorder
    tied observations and perturb an AUC in the sixth decimal.  Cheap
    reproducibility is worth more than the memory.
    """
    from .features import REQUIRED_PANEL_COLUMNS, build_features

    if refresh or not _PANEL_EXISTS():
        build_panel(refresh=refresh, save=True)

    frame = load_panel(columns=list(REQUIRED_PANEL_COLUMNS))
    frame = build_features(frame)
    frame = add_outcomes(frame)

    keep = [column for column in analysis_columns() if column in frame.columns]
    frame = _downcast(frame[keep].copy())

    if save:
        _DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(_DATASET_PATH, index=False)
    return frame


def _cache_is_stale(columns: Sequence[str] | None) -> bool:
    """Whether the cached analysis frame lacks a column the build now produces."""
    if not _DATASET_PATH.exists():
        return False
    import pyarrow.parquet as pq

    cached = set(pq.read_schema(_DATASET_PATH).names)
    expected = set(analysis_columns())
    if columns is not None:
        expected &= set(columns)
    missing = sorted(expected - cached)
    if missing:
        log.warning("cached analysis frame predates %s; rebuilding it", missing)
    return bool(missing)


def load_dataset(years: Iterable[int] | None = None,
                 columns: Sequence[str] | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """Load the cached analysis frame, building it on first use.

    Parameters
    ----------
    years:
        Restrict to these fiscal-period-end years.  Pushed down to the parquet
        reader, so unwanted row groups are never materialised.
    columns:
        Restrict to these columns.  ``bn`` and ``year`` are always included.

    A cache written before a feature was added lacks that column.  If any
    column the current build would produce is missing from the cache, the
    analysis frame is rebuilt from the cached panel (the raw CSVs are not
    re-read), rather than failing inside the parquet reader.
    """
    if not _DATASET_PATH.exists() or refresh or _cache_is_stale(columns):
        build_dataset(refresh=refresh)

    read_columns = None
    if columns is not None:
        read_columns = list(dict.fromkeys(["bn", "year", *columns]))

    filters = None
    if years is not None:
        filters = [("year", "in", list(years))]

    frame = pd.read_parquet(_DATASET_PATH, columns=read_columns, filters=filters)
    for column in CATEGORICAL:
        if column in frame.columns and not isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("category")
    return frame


def _split(frame: pd.DataFrame, target: str, train_year: int,
           test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = frame[target].notna()
    train = frame.loc[(frame["year"] == train_year) & labelled]
    test = frame.loc[(frame["year"] == test_year) & labelled]
    if train.empty or test.empty:
        raise ValueError(
            f"empty split for {target!r}: train={len(train)} rows in {train_year}, "
            f"test={len(test)} rows in {test_year}. Did you load those years?"
        )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def exit_split(frame: pd.DataFrame, train_year: int = TRAIN_YEAR,
               test_year: int = TEST_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split for the exit outcome.

    Year *t* features require year *t-1* for the growth terms, and the exit
    label requires at least two later years, which leaves 2020 and 2021 as the
    only fully-specified years in a 2019-2023 window.  Train on the earlier,
    test on the later.  The two samples share organisations but not periods:
    this is a forecasting test, not a held-out-organisations test, which is the
    right question for a monitoring tool that will be re-fitted every year.
    """
    return _split(frame, "exit_next_year", train_year, test_year)


def vulnerability_split(frame: pd.DataFrame, train_year: int = 2019,
                        test_year: int = 2020) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split for the Greenlee-Trussel net-asset-decline outcome.

    The three-year horizon means only 2019 (resolved in 2022) and 2020
    (resolved in 2023) can be labelled.  Growth features do not exist in 2019 —
    it is the first year of the window — so the training rows carry missing
    dynamics, which the pipelines handle by imputation-with-indicator or by
    splitting on missingness.
    """
    return _split(frame, "vulnerable", train_year, test_year)


def schedule_6_split(frame: pd.DataFrame, train_year: int = TRAIN_YEAR,
                     test_year: int = TEST_YEAR, target: str = "exit_next_year",
                     require_lagged_tier: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split restricted to charities filing the detailed return.

    A charity is included in year *t* if it filed on Schedule 6 in *t*.  That is
    the operationally right condition — at the moment you are scoring, either
    you have the detailed return in front of you or you do not — but it makes
    the sample a **selected** one, and the selection is on size: Schedule 6 is
    the form you file once you outgrow the small-charity thresholds.  Results on
    this sample are not comparable with the full-population results in
    ``03-model-results`` and should not be read as an improvement on them.

    Two consequences follow, and both are visible in
    ``notebooks/04-schedule-6-deep-dive``:

    * the exit base rate is lower here, because small charities dominate exits;
    * ``tier_downgrade`` becomes uninformative in the training year, since a
      charity that fell back to the short form in *t* is excluded from *t*.

    Set ``require_lagged_tier=True`` to additionally require Schedule 6 in
    *t-1*, which makes the year-on-year deltas well defined for every row at the
    cost of dropping charities that graduated onto the detailed form.  That
    option needs ``frame`` to contain the year before ``train_year``; it raises
    rather than silently emptying the training sample if it does not.
    """
    is_schedule_6 = frame["filer_tier"].eq(SCHEDULE_6).fillna(False)
    subset = frame.loc[is_schedule_6]

    if require_lagged_tier:
        present = set(frame["year"].unique())
        missing = [year - 1 for year in (train_year, test_year) if year - 1 not in present]
        if missing:
            raise ValueError(
                f"require_lagged_tier needs the preceding year(s) {missing} in `frame`; "
                f"loaded years are {sorted(present)}"
            )
        lagged = frame.loc[is_schedule_6, ["bn", "year"]].assign(year=lambda d: d["year"] + 1)
        keep = pd.MultiIndex.from_frame(subset[["bn", "year"]]).isin(
            pd.MultiIndex.from_frame(lagged)
        )
        subset = subset.loc[keep]

    return _split(subset, target, train_year, test_year)


def scoring_frame(frame: pd.DataFrame, year: int = SCORE_YEAR) -> pd.DataFrame:
    """The most recent cohort with features but no trustworthy label.

    Used to produce a forward-looking watchlist rather than to measure
    anything: the 2022 cohort's exits can only be checked against a single
    later year, which is not enough to distinguish a permanent exit from a late
    filing.
    """
    subset = frame.loc[frame["year"] == year]
    if subset.empty:
        raise ValueError(f"no rows for year {year}; loaded years are "
                         f"{sorted(frame['year'].unique())}")
    return subset.reset_index(drop=True)
