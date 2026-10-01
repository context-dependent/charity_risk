"""The HS risk index: five financial dimensions scored from the HS ratios.

The HS index is a separate index from Tuckman and Chang (1991), built on its own
ratio set.  It scores an organisation on five financial dimensions — solvency,
liquidity, profitability, efficiency and revenue — using the twelve ratios in
:data:`charity_risk.features.HS_RATIOS`, grouped as in
:func:`charity_risk.features.add_hs_ratios`:

=================  ===================================================  ======
Dimension          Ratios                                               Risky
=================  ===================================================  ======
Solvency           equity balance, net assets / assets, net assets /    low
                   liabilities
Liquidity          working capital / assets, working-capital months     low
Profitability      return on assets, surplus, operating margin, markup  low
Efficiency         administrative cost ratio                            low
Revenue            revenue growth volatility, revenue concentration     high
=================  ===================================================  ======

Each ratio is scored against the training sample, averaged within its
dimension, and the dimensions are averaged into the index, so every dimension
carries equal weight however many ratios describe it.  No weight is fitted.

The per-ratio score comes in two forms (``method``):

``"percentile"`` (default)
    The ratio's position in the training distribution, oriented so that 1 is
    the riskiest end.  Nothing is thrown away, so the index is continuous.
``"flags"``
    1 inside the ratio's risky tail (by default the riskiest fifth of the
    training sample), 0 outside.  The index is then the dimension-weighted
    share of flags.

The Tuckman-Chang score in :mod:`charity_risk.benchmarks` is a different index —
four ratios, one flag each, no dimensions — and is used in this project only as
a benchmark to compare the HS index against.  The two share four ratios and, on
those four, the same risky tails.

Missing ratios are not scored — a missing value is never a flag — and a
dimension with no scored ratio is left out of the average rather than counted as
zero; otherwise the index would partly measure filing tier, since the two
liquidity ratios exist only for Schedule 6 filers.  The administrative cost
ratio's risky tail is the *bottom* (spending with no slack to cut), although
``02-features-and-benchmarks`` finds it runs the other way on exit; pass
``directions`` to override any ratio's orientation.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from .features import HS_RATIOS

__all__ = [
    "HS_DIMENSIONS",
    "HS_RISK_DIRECTION",
    "HSRiskIndex",
]

log = logging.getLogger(__name__)

#: The five financial dimensions of the index and the ratios that describe each.
HS_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "solvency": ("equity_balance", "net_assets_to_assets", "equity_ratio"),
    "liquidity": ("liquidity_ratio", "working_capital_months"),
    "profitability": ("return_on_assets", "net_revenue", "operating_margin",
                      "operating_markup"),
    "efficiency": ("admin_cost_ratio",),
    "revenue": ("revenue_growth_volatility", "revenue_concentration"),
}

#: Which tail of each ratio is the risky one: ``-1`` the bottom, ``+1`` the top.
#: On the four ratios it shares with the Tuckman-Chang benchmark, the direction
#: is the same as :data:`charity_risk.benchmarks.TC_RISK_DIRECTION`.
HS_RISK_DIRECTION: dict[str, int] = {
    "equity_balance": -1,
    "net_assets_to_assets": -1,
    "equity_ratio": -1,
    "liquidity_ratio": -1,
    "working_capital_months": -1,
    "return_on_assets": -1,
    "net_revenue": -1,
    "operating_margin": -1,
    "operating_markup": -1,
    "admin_cost_ratio": -1,
    "revenue_growth_volatility": +1,
    "revenue_concentration": +1,
}

assert {r for ratios in HS_DIMENSIONS.values() for r in ratios} == set(HS_RATIOS)
assert set(HS_RISK_DIRECTION) == set(HS_RATIOS)


class HSRiskIndex(BaseEstimator, ClassifierMixin):
    """Equal-weight index of per-dimension ratio risk.

    .. math::

        s_{k,i} = \\begin{cases}
            \\hat F_k(d_k\\, r_{k,i}) & \\text{percentile} \\\\
            \\mathbb{1}\\!\\left[r_{k,i} \\text{ in the risky tail of ratio } k\\right]
                                    & \\text{flags}
        \\end{cases}
        \\qquad
        D_{j,i} = \\operatorname{mean}_{k \\in j}\\, s_{k,i}
        \\qquad
        I_i = \\operatorname{mean}_{j}\\, D_{j,i}

    where :math:`d_k = \\pm 1` orients ratio :math:`k` so that larger is riskier,
    :math:`\\hat F_k` is its mid-rank empirical CDF in the training sample, and
    each mean runs over the ratios (dimensions) that are observed for
    organisation :math:`i`.  In plain terms: for every ratio, ask what share of
    last year's charities looked safer; average those shares within each of the
    five dimensions; then average the five.  An index of 0.8 means the charity
    sits, on average across dimensions, at the riskiest fifth of the sector.

    No weight is fitted.  :meth:`predict_proba` passes the index through a
    one-variable logit fitted on the training sample, which is monotone and so
    changes no ranking; it exists only so that the index can be scored on the
    same probability metrics as everything else.

    Parameters
    ----------
    method:
        ``"percentile"`` or ``"flags"``; see the module docstring.
    quantile:
        Tail mass flagged per ratio when ``method="flags"``.  Under the percentile method it
        only sets the default threshold of :meth:`dimension_flags`.
    dimensions:
        Dimension -> ratios.  Defaults to :data:`HS_DIMENSIONS`.
    directions:
        Ratio -> ``+1``/``-1`` overrides, merged over :data:`HS_RISK_DIRECTION`.

    Attributes set by :meth:`fit`
    -----------------------------
    reference_:
        Ratio -> sorted training values, for the percentile score.
    cutpoints_:
        Ratio -> the risky-quantile threshold, for the flag score.
    unscored_ratios_:
        Ratios entirely missing in the training sample, which cannot be scored.
        ``revenue_growth_volatility`` is one in a 2020 training year, since it
        needs two prior growth observations.
    calibrator_:
        The monotone logit from index to probability (only when ``y`` is given).
    """

    def __init__(self, method: str = "percentile", quantile: float = 0.2,
                 dimensions: Mapping[str, tuple[str, ...]] | None = None,
                 directions: Mapping[str, int] | None = None):
        self.method = method
        self.quantile = quantile
        self.dimensions = dimensions
        self.directions = directions

    # ------------------------------------------------------------ fitting
    def _resolved(self) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
        dimensions = dict(HS_DIMENSIONS if self.dimensions is None else self.dimensions)
        directions = {**HS_RISK_DIRECTION, **(self.directions or {})}
        return dimensions, directions

    def fit(self, X: pd.DataFrame, y=None) -> "HSRiskIndex":
        if self.method not in ("percentile", "flags"):
            raise ValueError(f"method must be 'percentile' or 'flags', not {self.method!r}")
        self.dimensions_, self.directions_ = self._resolved()
        self.ratios_ = [r for ratios in self.dimensions_.values() for r in ratios]

        self.reference_: dict[str, np.ndarray] = {}
        self.cutpoints_: dict[str, float] = {}
        self.unscored_ratios_: list[str] = []
        for ratio in self.ratios_:
            values = pd.to_numeric(X[ratio], errors="coerce").to_numpy(dtype="float64")
            values = np.sort(values[np.isfinite(values)])
            if values.size == 0:
                self.unscored_ratios_.append(ratio)
                continue
            self.reference_[ratio] = values
            q = self.quantile if self.directions_[ratio] < 0 else 1.0 - self.quantile
            self.cutpoints_[ratio] = float(np.quantile(values, q))
        if self.unscored_ratios_:
            log.warning("HS index: ratios entirely missing in the training sample "
                        "are not scored: %s", self.unscored_ratios_)

        self.calibrator_ = None
        if y is not None:
            y = np.asarray(y, dtype="float64")
            index = self.index(X).to_numpy()
            usable = np.isfinite(index) & np.isfinite(y)
            self.classes_ = np.unique(y[np.isfinite(y)])
            self.base_rate_ = float(y[np.isfinite(y)].mean())
            self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(
                index[usable].reshape(-1, 1), y[usable])
        return self

    # ------------------------------------------------------------ scoring
    def _percentile(self, ratio: str, values: np.ndarray) -> np.ndarray:
        reference = self.reference_[ratio]
        below = np.searchsorted(reference, values, side="left")
        at_or_below = np.searchsorted(reference, values, side="right")
        cdf = (below + at_or_below) / (2.0 * reference.size)
        return cdf if self.directions_[ratio] > 0 else 1.0 - cdf

    def _flag(self, ratio: str, values: np.ndarray) -> np.ndarray:
        cut = self.cutpoints_[ratio]
        flagged = values <= cut if self.directions_[ratio] < 0 else values >= cut
        return flagged.astype("float64")

    def ratio_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-ratio risk in ``[0, 1]`` (``NaN`` where the ratio is missing)."""
        score = self._percentile if self.method == "percentile" else self._flag
        out = {}
        for ratio in self.ratios_:
            values = pd.to_numeric(X[ratio], errors="coerce").to_numpy(dtype="float64")
            if ratio in self.unscored_ratios_:
                out[ratio] = np.full(len(values), np.nan)
                continue
            scored = score(ratio, values)
            out[ratio] = np.where(np.isfinite(values), scored, np.nan)
        return pd.DataFrame(out, index=X.index)

    def dimension_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean ratio risk within each dimension (``NaN`` if none observed)."""
        ratios = self.ratio_scores(X)
        return pd.DataFrame({
            dimension: ratios[list(members)].mean(axis=1, skipna=True)
            for dimension, members in self.dimensions_.items()
        }, index=X.index)

    def index(self, X: pd.DataFrame) -> pd.Series:
        """The index in ``[0, 1]``: the mean of the observed dimension scores."""
        return self.dimension_scores(X).mean(axis=1, skipna=True).rename("hs_index")

    def n_dimensions_scored(self, X: pd.DataFrame) -> pd.Series:
        return self.dimension_scores(X).notna().sum(axis=1)

    def dimension_flags(self, X: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """Whether each dimension is at risk (``NaN`` where it is unobserved).

        A dimension is at risk when its score reaches ``threshold``, which
        defaults to ``1 - quantile`` under the percentile method (on average in
        the risky tail) and ``0.5`` under the flag method (at least half its
        ratios flagged).
        """
        if threshold is None:
            threshold = 1.0 - self.quantile if self.method == "percentile" else 0.5
        scores = self.dimension_scores(X)
        return (scores >= threshold).astype("float64").where(scores.notna())

    def n_dimensions_at_risk(self, X: pd.DataFrame, threshold: float | None = None) -> pd.Series:
        """How many of the five dimensions are at risk (0-5)."""
        return self.dimension_flags(X, threshold).sum(axis=1, min_count=1).fillna(0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Two-column probability array from the monotone calibration logit.

        Organisations with no scored dimension get the training base rate.
        """
        if self.calibrator_ is None:
            raise RuntimeError("fit with a `y` argument before calling predict_proba")
        index = self.index(X).to_numpy()
        positive = np.full(len(index), self.base_rate_)
        observed = np.isfinite(index)
        if observed.any():
            positive[observed] = self.calibrator_.predict_proba(
                index[observed].reshape(-1, 1))[:, 1]
        return np.column_stack([1.0 - positive, positive])
