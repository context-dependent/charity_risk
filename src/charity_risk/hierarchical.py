"""A partially-pooled Bayesian logistic model of charity exit.

Why bother, when gradient boosting already ranks better?  Because ranking is
not the only thing a supervisor wants from a risk model.  Two questions the
point-estimate models cannot answer:

1. *How much of the variation in exit risk is between sectors and provinces
   rather than between organisations?*  The hierarchical variance parameters
   answer that directly.
2. *How confident should I be about this particular charity's score?*  A
   posterior gives an interval; a boosted tree gives a number.

The specification is a logistic regression with non-centred hierarchical
intercepts for charity category and for province, crossed:

.. math::

    \\operatorname{logit} p_i = \\alpha + u_{c[i]} + v_{p[i]} + x_i^\\top \\beta

with :math:`u_c \\sim \\mathcal{N}(0, \\sigma_u^2)` and
:math:`v_p \\sim \\mathcal{N}(0, \\sigma_v^2)`.  Categories with few charities
are shrunk towards the national average, which is exactly what one wants from a
screening tool: a category with eleven registrants should not get an extreme
baseline hazard on the strength of one exit.

Inference defaults to ADVI, which fits in a couple of minutes on 84,000 rows.
``method="nuts"`` gives the honest posterior and takes considerably longer;
the variational fit is close enough for the group-level variance comparisons
made in the notebooks, and both are exposed so the difference can be checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

__all__ = [
    "HIERARCHICAL_FEATURES",
    "HierarchicalFit",
    "fit_hierarchical",
    "group_effect_table",
]

#: A compact design.  The hierarchical model exists for interpretation, so it
#: uses ratios with a direction the literature has an opinion about rather than
#: the full extended set.
HIERARCHICAL_FEATURES: tuple[str, ...] = (
    "log_total_revenue",
    "log_age",
    "equity_balance",
    "revenue_concentration",
    "admin_cost_ratio",
    "operating_margin",
    "debt_ratio",
    "months_of_cash",
    "revenue_growth",
    "donation_dependence",
    "share_government",
    "consecutive_deficit_years",
    "is_section_d",
)


@dataclass
class HierarchicalFit:
    """A fitted hierarchical model plus everything needed to score new rows."""

    idata: object
    features: tuple[str, ...]
    means: pd.Series
    stds: pd.Series
    lower: pd.Series
    upper: pd.Series
    category_levels: pd.Index
    province_levels: pd.Index
    method: str

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        design = frame[list(self.features)].astype("float64")
        design = design.clip(lower=self.lower, upper=self.upper, axis=1)
        design = design.fillna(self.means)
        return ((design - self.means) / self.stds).to_numpy()

    def predict(self, frame: pd.DataFrame, n_samples: int = 400,
                chunk_size: int = 8_192,
                random_state: int = RANDOM_STATE) -> pd.DataFrame:
        """Posterior mean exit probability and an 80% credible interval.

        Unseen category or province levels fall back to the pooled intercept,
        which is the correct behaviour for a new sector rather than an error.

        Scored in row chunks: the full ``n_rows x n_samples`` probability matrix
        for a year of the register is around a gigabyte, and three intermediates
        of that size exist at once in the naive version.
        """
        import arviz as az

        posterior = az.extract(self.idata, group="posterior",
                               num_samples=n_samples, random_seed=random_state)
        alpha = np.asarray(posterior["alpha"], dtype="float32")
        beta = np.asarray(posterior["beta"], dtype="float32")
        u = np.asarray(posterior["u_category"], dtype="float32")
        v = np.asarray(posterior["v_province"], dtype="float32")

        X = self._design(frame).astype("float32")
        cat_idx = self.category_levels.get_indexer(frame["category_desc"].astype("string"))
        prov_idx = self.province_levels.get_indexer(frame["province"].astype("string"))

        n_rows = len(frame)
        mean = np.empty(n_rows, dtype="float64")
        q10 = np.empty(n_rows, dtype="float64")
        q90 = np.empty(n_rows, dtype="float64")

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            cat = cat_idx[start:stop]
            prov = prov_idx[start:stop]
            eta = alpha[None, :] + X[start:stop] @ beta
            eta += np.where(cat[:, None] >= 0, u[np.clip(cat, 0, None), :], 0.0)
            eta += np.where(prov[:, None] >= 0, v[np.clip(prov, 0, None), :], 0.0)
            probability = 1.0 / (1.0 + np.exp(-eta, dtype="float32"))
            mean[start:stop] = probability.mean(axis=1)
            q10[start:stop], q90[start:stop] = np.quantile(probability, [0.10, 0.90], axis=1)

        return pd.DataFrame({"risk_mean": mean, "risk_q10": q10, "risk_q90": q90},
                            index=frame.index)


def fit_hierarchical(train: pd.DataFrame, target: str = "exit_next_year",
                     features: Sequence[str] = HIERARCHICAL_FEATURES,
                     method: str = "advi", draws: int = 1000, tune: int = 1000,
                     chains: int = 2, advi_iterations: int = 30_000,
                     winsor: tuple[float, float] = (0.01, 0.99),
                     random_state: int = RANDOM_STATE) -> HierarchicalFit:
    """Fit the crossed hierarchical logit.

    Predictors are winsorised at ``winsor`` and standardised, with missing
    values set to the (winsorised) mean.  Mean-imputation is cruder than what
    the scikit-learn pipelines do, and it is used here deliberately: the point
    of this model is the interpretability of :math:`\\beta`, which a
    missing-indicator interaction would muddy.  Missingness is concentrated in
    the short-form filers, and ``is_section_d`` is in the design to absorb it.

    Parameters
    ----------
    method:
        ``"advi"`` for mean-field variational inference (minutes) or ``"nuts"``
        for MCMC (tens of minutes to hours at this sample size).
    """
    import pymc as pm

    data = train.loc[train[target].notna()].copy()
    y = data[target].astype("float64").to_numpy()

    design = data[list(features)].astype("float64")
    lower = design.quantile(winsor[0])
    upper = design.quantile(winsor[1])
    design = design.clip(lower=lower, upper=upper, axis=1)
    means = design.mean()
    stds = design.std().replace(0.0, 1.0)
    X = ((design.fillna(means) - means) / stds).to_numpy()

    category = data["category_desc"].astype("string").fillna("unknown")
    province = data["province"].astype("string").fillna("unknown")
    category_levels = pd.Index(sorted(category.unique()))
    province_levels = pd.Index(sorted(province.unique()))
    cat_idx = category_levels.get_indexer(category)
    prov_idx = province_levels.get_indexer(province)

    coords = {
        "feature": list(features),
        "category": list(category_levels),
        "province": list(province_levels),
    }

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=-4.0, sigma=2.0)  # ~2% base rate on the logit scale
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")

        sigma_u = pm.HalfNormal("sigma_category", sigma=1.0)
        sigma_v = pm.HalfNormal("sigma_province", sigma=1.0)
        u_raw = pm.Normal("u_category_raw", 0.0, 1.0, dims="category")
        v_raw = pm.Normal("v_province_raw", 0.0, 1.0, dims="province")
        u = pm.Deterministic("u_category", u_raw * sigma_u, dims="category")
        v = pm.Deterministic("v_province", v_raw * sigma_v, dims="province")

        eta = alpha + pm.math.dot(X, beta) + u[cat_idx] + v[prov_idx]
        pm.Bernoulli("y", logit_p=eta, observed=y)

        if method == "advi":
            approximation = pm.fit(n=advi_iterations, method="advi",
                                   random_seed=random_state, progressbar=False)
            idata = approximation.sample(draws, random_seed=random_state)
        elif method == "nuts":
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              random_seed=random_state, progressbar=False,
                              target_accept=0.9)
        else:
            raise ValueError(f"unknown method {method!r}; use 'advi' or 'nuts'")

    return HierarchicalFit(idata=idata, features=tuple(features), means=means, stds=stds,
                           lower=lower, upper=upper, category_levels=category_levels,
                           province_levels=province_levels, method=method)


def group_effect_table(fit: HierarchicalFit, group: str = "category") -> pd.DataFrame:
    """Posterior summary of one set of group intercepts, on the log-odds scale.

    Positive values mean a higher baseline exit hazard than the national
    average, holding the financial ratios fixed.
    """
    import arviz as az

    variable = "u_category" if group == "category" else "v_province"
    summary = az.summary(fit.idata, var_names=[variable], ci_prob=0.9)
    summary.index = summary.index.astype(str).str.replace(
        rf"^{variable}\[|\]$", "", regex=True
    )
    return summary.sort_values("mean", ascending=False)
