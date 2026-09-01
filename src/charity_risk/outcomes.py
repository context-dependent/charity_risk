"""Outcome definitions: what counts as failure.

Two labels are constructed, because "corporate failure" in the charity sector
has no single observable analogue and the literature we benchmark against uses
the weaker of the two.

``exit``
    The organisation stops filing the T3010 permanently.  A registered Canadian
    charity must file within six months of its fiscal period end for as long as
    it is registered; ceasing to appear therefore means the registration ended
    — voluntary revocation, revocation for cause, revocation for failure to
    file, amalgamation or wind-up.  This is the closest available analogue to
    corporate failure, and it is the primary outcome of this project.

``vulnerable``
    Net assets fall by at least 20% over three years.  This is the operational
    definition used by Greenlee and Trussel (2000) and Trussel (2002), inherited
    from Tuckman and Chang (1991).  It is a *financial distress* measure, not a
    failure measure, and is included so that our benchmark replication is
    faithful to the original studies.

Two things about the exit label deserve to be stated plainly rather than buried:

* **Disappearance is not always failure.**  A charity that merges into another,
  or that winds up solvent and by choice, exits the panel exactly as one that
  collapses does.  The label is a proxy; every result should be read as
  "predicts deregistration", not "predicts insolvency".
* **Right-censoring.**  A charity absent from year *t+1* may simply be filing
  late relative to the extract date.  The panel does contain gap-then-return
  patterns (about 0.5% of BNs).  We therefore require
  :data:`~charity_risk.config.MIN_LOOKAHEAD_YEARS` observed years after *t*
  before we are willing to call an absence permanent, which restricts labelled
  years to 2019-2021 in a 2019-2023 window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_LOOKAHEAD_YEARS, PANEL_END, YEARS

__all__ = [
    "add_exit_labels",
    "add_vulnerability_labels",
    "add_outcomes",
    "labelled_years",
    "outcome_summary",
]


def labelled_years(panel_end: int = PANEL_END, years: tuple[int, ...] = YEARS,
                   min_lookahead: int = MIN_LOOKAHEAD_YEARS) -> list[int]:
    """Years for which the permanent-exit label is trustworthy."""
    return [year for year in years if panel_end - year >= min_lookahead]


def add_exit_labels(panel: pd.DataFrame, years: tuple[int, ...] = YEARS,
                    min_lookahead: int = MIN_LOOKAHEAD_YEARS) -> pd.DataFrame:
    """Attach the permanent-exit label and its supporting columns.

    Adds
    ------
    ``exit_next_year``
        1 if the charity filed in ``year`` and never files again within the
        observation window; ``NaN`` when the remaining lookahead is shorter
        than ``min_lookahead``.
    ``exit_provisional``
        The same rule without the lookahead requirement.  Useful for describing
        the final year of the window, but too noisy to model on: roughly one in
        eight single-year absences is followed by a return.
    ``years_observed_after``
        How many later years of the window exist, i.e. the lookahead depth.
    ``survives_window``
        1 if the charity is still filing in the final year of the window.
    """
    panel = panel.copy()
    panel_end = max(years)

    panel["years_observed_after"] = (panel_end - panel["year"]).astype("int16")
    panel["exit_provisional"] = (panel["last_filing_year"] == panel["year"]).astype("int8")

    trustworthy = panel["years_observed_after"] >= min_lookahead
    panel["exit_next_year"] = np.where(trustworthy, panel["exit_provisional"], np.nan)
    panel["survives_window"] = (panel["last_filing_year"] == panel_end).astype("int8")
    return panel


def _net_assets_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="bn", columns="year", values="net_assets", aggfunc="last")


def add_vulnerability_labels(panel: pd.DataFrame, horizon: int = 3,
                             threshold: float = 0.20) -> pd.DataFrame:
    """Attach the Greenlee-Trussel net-asset-decline label.

    ``vulnerable`` is 1 when net assets at ``year + horizon`` are at least
    ``threshold`` below net assets at ``year``.

    Following Trussel (2002), organisations with non-positive net assets in the
    base year are excluded (the percentage change is not interpretable), as are
    those without a balance sheet at both ends of the horizon.  Those cases get
    ``NaN``, not 0.

    A charity that exits before ``year + horizon`` has no terminal net-asset
    figure.  Coding it as "not vulnerable" would be perverse, so it is coded
    ``NaN`` here and handled explicitly in
    :func:`charity_risk.evaluate.compare_outcomes`; the exit label is the right
    instrument for those organisations.
    """
    panel = panel.copy()
    wide = _net_assets_by_year(panel)

    base = panel.set_index(["bn", "year"]).index
    start = pd.Series(panel["net_assets"].to_numpy(), index=base)

    future_year = panel["year"] + horizon
    lookup = pd.MultiIndex.from_arrays([panel["bn"].to_numpy(), future_year.to_numpy()])
    stacked = wide.stack()
    end = stacked.reindex(lookup).to_numpy()

    start_values = start.to_numpy()
    valid = (start_values > 0) & np.isfinite(end) & np.isfinite(start_values)
    change = np.divide(end - start_values, np.abs(start_values),
                       out=np.full(start_values.shape, np.nan), where=valid)

    panel["net_asset_change_3y"] = change
    panel["vulnerable"] = np.where(valid, (change <= -threshold).astype(float), np.nan)
    panel["vulnerable_horizon_observable"] = future_year.isin(panel["year"].unique()).astype("int8")
    return panel


def add_outcomes(panel: pd.DataFrame, years: tuple[int, ...] = YEARS,
                 min_lookahead: int = MIN_LOOKAHEAD_YEARS,
                 horizon: int = 3, threshold: float = 0.20) -> pd.DataFrame:
    """Apply both outcome definitions."""
    panel = add_exit_labels(panel, years=years, min_lookahead=min_lookahead)
    return add_vulnerability_labels(panel, horizon=horizon, threshold=threshold)


def outcome_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-year base rates and sample sizes for both outcomes."""
    grouped = panel.groupby("year", observed=True)
    summary = pd.DataFrame({
        "n_filers": grouped.size(),
        "n_exit_labelled": grouped["exit_next_year"].count(),
        "exit_rate": grouped["exit_next_year"].mean(),
        "exit_rate_provisional": grouped["exit_provisional"].mean(),
        "n_vulnerable_labelled": grouped["vulnerable"].count(),
        "vulnerable_rate": grouped["vulnerable"].mean(),
    })
    return summary
