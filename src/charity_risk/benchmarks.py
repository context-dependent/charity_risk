"""The classical benchmarks: Tuckman-Chang (1991) and its logit descendants.

Three reference models are reproduced here as faithfully as the Canadian T3010
allows:

**Tuckman and Chang (1991)** is not a fitted model at all.  It is a scoring
rule: compute four ratios, flag an organisation on each ratio if it falls in
the risky quintile of the sector, and count the flags.  Zero flags is healthy;
four is "severely at risk".  :class:`TuckmanChangScore` implements exactly
that, with the quintile cut-points *fitted on a training sample* and applied
unchanged out of sample, so that the benchmark faces the same information
constraint as everything it is compared against.

**Greenlee and Trussel (2000)** put the same four ratios into a logistic
regression of financial vulnerability.  That is :data:`GREENLEE_TRUSSEL_2000`,
fitted through :func:`charity_risk.models.build_logit_pipeline` — see that
function's docstring for the formal specification common to both logits below.

**Trussel (2002)** revised the specification to four ratios — revenue
concentration, surplus margin, debt ratio and administrative cost ratio — plus
a size control and sector fixed effects.  That is :data:`TRUSSEL_2002`.

Where the original studies used US Form 990 data, the T3010 line that most
nearly corresponds is used; the mapping is documented in
:func:`charity_risk.features.add_tuckman_chang`.  The one substantive
divergence is revenue concentration, which we compute over eight harmonised
buckets so that short-form and long-form filers are comparable — see
:data:`charity_risk.fields.REVENUE_SOURCES_CORE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import TRUSSEL_CONTROLS, TRUSSEL_RATIOS, TUCKMAN_CHANG

__all__ = [
    "TC_RISK_DIRECTION",
    "TuckmanChangScore",
    "GREENLEE_TRUSSEL_2000",
    "TRUSSEL_2002",
    "fit_statsmodels_logit",
]

#: Which tail of each Tuckman-Chang ratio is the risky one.  ``-1`` means the
#: bottom quintile is risky, ``+1`` the top quintile.  The administrative cost
#: ratio is ``-1`` because Tuckman and Chang treat administrative spending as
#: slack that can be cut in a shock: a charity with almost none has nothing to
#: give up.
TC_RISK_DIRECTION: dict[str, int] = {
    "equity_balance": -1,
    "revenue_concentration": +1,
    "admin_cost_ratio": -1,
    "operating_margin": -1,
}


@dataclass
class TuckmanChangScore:
    """The Tuckman-Chang 0-4 vulnerability count.

    .. math::

        S_i = \\sum_{k=1}^{4} \\mathbb{1}\\!\\left[r_{k,i} \\text{ in the sector's
        risky quintile of ratio } k\\right]

    where each :math:`r_{k,i}` is one of the four ratios and the quintile
    cut-point for each is estimated once, on the training sample, and then
    applied unchanged.  In plain terms: not a fitted model at all, but a
    checklist — flag an organisation on a ratio if it sits in the worst 20% of
    the sector on that ratio, and count how many of the four checkboxes are
    ticked.  :meth:`predict_proba` turns the resulting 0-4 count into a
    probability only by looking up each count's historical exit rate, which is
    why the rule can only ever produce five distinct risk scores.

    Parameters
    ----------
    quantile:
        Tail mass treated as "at risk" on each ratio.  ``0.2`` reproduces the
        original quintile rule.
    ratios:
        Which ratios to score.  Defaults to all four.

    Attributes set by :meth:`fit`
    -----------------------------
    cutpoints_:
        Ratio -> threshold, estimated on the training sample.
    risk_rate_by_score_:
        Score (0-4) -> observed outcome rate in the training sample.  Used by
        :meth:`predict_proba` so that a rule-based count can be compared with
        fitted models on probability metrics such as the Brier score.

    Notes
    -----
    Missing ratios do not count as flags.  A Section D filer with no
    administrative-cost line is scored on the three ratios it does have, and
    ``n_ratios_scored`` records how many contributed.  Counting missingness as
    a flag would make the score a proxy for filing tier.
    """

    quantile: float = 0.2
    ratios: tuple[str, ...] = TUCKMAN_CHANG
    cutpoints_: dict[str, float] = field(default_factory=dict, init=False)
    risk_rate_by_score_: dict[int, float] = field(default_factory=dict, init=False)
    base_rate_: float = field(default=np.nan, init=False)

    def fit(self, frame: pd.DataFrame, y: pd.Series | None = None) -> "TuckmanChangScore":
        for ratio in self.ratios:
            direction = TC_RISK_DIRECTION[ratio]
            q = self.quantile if direction < 0 else 1.0 - self.quantile
            self.cutpoints_[ratio] = float(frame[ratio].quantile(q))

        if y is not None:
            scores = self.score(frame)
            labelled = pd.notna(y)
            self.base_rate_ = float(np.asarray(y)[labelled].mean())
            grouped = pd.DataFrame({"score": scores[labelled], "y": np.asarray(y)[labelled]})
            self.risk_rate_by_score_ = grouped.groupby("score")["y"].mean().to_dict()
        return self

    def flags(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-ratio at-risk indicators (``NaN`` where the ratio is missing)."""
        out = {}
        for ratio in self.ratios:
            values = frame[ratio]
            cut = self.cutpoints_[ratio]
            flag = values <= cut if TC_RISK_DIRECTION[ratio] < 0 else values >= cut
            out[f"tc_flag_{ratio}"] = flag.astype("float64").where(values.notna())
        return pd.DataFrame(out, index=frame.index)

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """The 0-4 count of at-risk ratios."""
        return self.flags(frame).sum(axis=1, min_count=1).fillna(0.0)

    def n_ratios_scored(self, frame: pd.DataFrame) -> pd.Series:
        return self.flags(frame).notna().sum(axis=1)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Two-column probability array, calibrated from the training sample.

        Every organisation with the same count gets the same probability, so
        the resulting ranking has exactly five levels.  That coarseness is the
        method, not an implementation shortcut, and it is the main reason the
        rule loses to a fitted model on ranking metrics.
        """
        if not self.risk_rate_by_score_:
            raise RuntimeError("fit with a `y` argument before calling predict_proba")
        scores = self.score(frame)
        positive = scores.map(self.risk_rate_by_score_).fillna(self.base_rate_).to_numpy()
        return np.column_stack([1.0 - positive, positive])


#: Greenlee and Trussel (2000): logistic regression on the four Tuckman-Chang
#: ratios, no controls.
GREENLEE_TRUSSEL_2000: dict[str, tuple[str, ...]] = {
    "numeric": TUCKMAN_CHANG,
    "categorical": (),
}

#: Trussel (2002): four ratios, a size control and sector fixed effects.
TRUSSEL_2002: dict[str, tuple[str, ...]] = {
    "numeric": TRUSSEL_RATIOS + TRUSSEL_CONTROLS,
    "categorical": ("charity_type",),
}


def fit_statsmodels_logit(frame: pd.DataFrame, target: str, numeric: tuple[str, ...],
                          categorical: tuple[str, ...] = (),
                          winsor: tuple[float, float] = (0.01, 0.99)):
    """Fit a benchmark logit with statsmodels, for a publishable coefficient table.

    The scoring path in :mod:`charity_risk.models` uses scikit-learn; this is a
    parallel fit whose purpose is inference — signs, magnitudes and standard
    errors comparable with the tables in the source papers.  Rows with any
    missing ratio are dropped (listwise deletion, as in the originals) and
    ratios are winsorised at the given quantiles, without which a handful of
    charity-years with near-zero revenue dominate the likelihood.

    Returns the fitted ``statsmodels`` results object; ``.summary2()`` gives the
    table.
    """
    import statsmodels.api as sm

    columns = list(numeric) + list(categorical) + [target]
    data = frame[columns].dropna()

    design = data[list(numeric)].astype("float64")
    lower = design.quantile(winsor[0])
    upper = design.quantile(winsor[1])
    design = design.clip(lower=lower, upper=upper, axis=1)

    if categorical:
        dummies = pd.get_dummies(data[list(categorical)].astype("string"),
                                 drop_first=True, dtype=float)
        design = pd.concat([design, dummies], axis=1)

    design = sm.add_constant(design, has_constant="add")
    model = sm.Logit(data[target].astype("float64"), design)
    return model.fit(disp=False, maxiter=200)
