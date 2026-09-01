"""Fitting, scoring and comparison of the model set.

The evaluation is deliberately unforgiving about two things.

**Temporal honesty.**  Models are fitted on one year and scored on a strictly
later one.  Nothing about the test year — not a quantile, not a category level,
not an imputation median — is available at fit time, because every such
quantity is learned inside the pipeline.

**Metrics matched to the decision.**  The exit base rate is around 2%, so
accuracy is meaningless and ROC-AUC is optimistic about a screening tool's
usefulness.  The headline numbers are therefore average precision (area under
the precision-recall curve) and *lift in the top 1% and 5% of the risk
ranking*, which is what a regulator or funder with a fixed review budget
actually experiences.  Calibration is reported separately, because a
well-ranked but badly calibrated score is fine for triage and useless for
expected-loss arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .benchmarks import TuckmanChangScore
from .config import RANDOM_STATE
from .models import SPECIFICATIONS, ModelSpec

__all__ = [
    "FittedModel",
    "fit_spec",
    "fit_all",
    "score_metrics",
    "lift_table",
    "calibration_table",
    "compare_models",
    "bootstrap_auc_difference",
    "bootstrap_metric_difference",
    "permutation_importance_table",
]

log = logging.getLogger(__name__)

_TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


@dataclass
class FittedModel:
    """A fitted specification plus the rows it was fitted on."""

    spec: ModelSpec
    estimator: object
    n_train: int
    train_base_rate: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.spec.family == "rule":
            return self.estimator.predict_proba(frame[list(self.spec.numeric)])
        return self.estimator.predict_proba(frame[self.spec.columns])

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(frame)[:, 1]


def _labelled(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame.loc[frame[target].notna()]


def fit_spec(spec: ModelSpec, train: pd.DataFrame, target: str) -> FittedModel:
    """Fit one specification on ``train``.

    Rows with a missing label are dropped.  Rows with missing *features* are
    kept — handling them is the pipeline's job, and dropping them would quietly
    change the estimation sample between specifications and make the comparison
    incoherent.
    """
    data = _labelled(train, target)
    y = data[target].astype("float64").to_numpy()

    unusable = [column for column in spec.numeric if data[column].notna().sum() == 0]
    if unusable:
        log.warning("%s: features entirely missing in the training sample, "
                    "neutralised to zero: %s", spec.name, unusable)

    if spec.family == "rule":
        estimator = TuckmanChangScore(ratios=spec.numeric).fit(data[list(spec.numeric)], y)
    else:
        estimator = spec.build()
        estimator.fit(data[spec.columns], y)

    return FittedModel(spec=spec, estimator=estimator, n_train=len(data),
                       train_base_rate=float(y.mean()))


def fit_all(train: pd.DataFrame, target: str,
            specs: dict[str, ModelSpec] | None = None) -> dict[str, FittedModel]:
    """Fit every specification in the registry."""
    specs = SPECIFICATIONS if specs is None else specs
    return {name: fit_spec(spec, train, target) for name, spec in specs.items()}


def _top_fraction_stats(y_true: np.ndarray, risk: np.ndarray,
                        fractions: tuple[float, ...] = _TOP_FRACTIONS) -> dict[str, float]:
    """Precision, recall and lift within the highest-risk ``k`` of the ranking.

    Ties are broken by a fixed random jitter rather than by input order, so a
    coarse score such as the 0-4 Tuckman-Chang count is measured on the average
    of its tied group rather than on whichever rows happen to sort first.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    order = np.lexsort((rng.random(len(risk)), -risk))
    ranked = y_true[order]
    base = y_true.mean()
    stats: dict[str, float] = {}
    for fraction in fractions:
        k = max(1, int(round(fraction * len(ranked))))
        captured = ranked[:k].sum()
        precision = captured / k
        stats[f"precision_top{int(fraction * 100)}pct"] = float(precision)
        stats[f"recall_top{int(fraction * 100)}pct"] = float(captured / max(y_true.sum(), 1))
        stats[f"lift_top{int(fraction * 100)}pct"] = float(precision / base) if base > 0 else np.nan
    return stats


def _calibration_fit(y_true: np.ndarray, risk: np.ndarray) -> tuple[float, float]:
    """Calibration intercept and slope from a logit-on-logit recalibration.

    A perfectly calibrated score gives intercept 0 and slope 1.  Slope below 1
    means the score is over-dispersed (too confident at both ends).
    """
    import statsmodels.api as sm

    eps = 1e-6
    p = np.clip(risk, eps, 1 - eps)
    design = sm.add_constant(np.log(p / (1 - p)), has_constant="add")
    try:
        result = sm.Logit(y_true, design).fit(disp=False, maxiter=100)
    except Exception:  # separation or singular design
        return (np.nan, np.nan)
    return float(result.params[0]), float(result.params[1])


def score_metrics(y_true: np.ndarray, risk: np.ndarray, calibration: bool = True) -> dict[str, float]:
    """Discrimination, calibration and top-of-ranking metrics for one model."""
    y_true = np.asarray(y_true, dtype="float64")
    risk = np.asarray(risk, dtype="float64")
    metrics: dict[str, float] = {
        "n": float(len(y_true)),
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, risk)),
        "average_precision": float(average_precision_score(y_true, risk)),
        "brier": float(brier_score_loss(y_true, risk)),
        "log_loss": float(log_loss(y_true, np.clip(risk, 1e-9, 1 - 1e-9))),
    }
    # Brier skill against the constant base-rate forecast: 0 = no better than
    # predicting the average, 1 = perfect.
    base = y_true.mean()
    reference = brier_score_loss(y_true, np.full_like(y_true, base))
    metrics["brier_skill"] = float(1.0 - metrics["brier"] / reference) if reference > 0 else np.nan
    metrics.update(_top_fraction_stats(y_true, risk))
    if calibration:
        intercept, slope = _calibration_fit(y_true, risk)
        metrics["calibration_intercept"] = intercept
        metrics["calibration_slope"] = slope
    return metrics


def lift_table(y_true: np.ndarray, risk: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: outcome rate and lift by risk decile, worst risk first."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype="float64"),
                          "risk": np.asarray(risk, dtype="float64")})
    frame["decile"] = pd.qcut(frame["risk"].rank(method="first", ascending=False),
                              q=n_bins, labels=range(1, n_bins + 1))
    grouped = frame.groupby("decile", observed=True)
    table = pd.DataFrame({
        "n": grouped.size(),
        "n_events": grouped["y"].sum(),
        "event_rate": grouped["y"].mean(),
        "mean_predicted": grouped["risk"].mean(),
    })
    table["lift"] = table["event_rate"] / frame["y"].mean()
    table["cumulative_recall"] = table["n_events"].cumsum() / table["n_events"].sum()
    return table


def calibration_table(y_true: np.ndarray, risk: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rates in equal-count bins of predicted risk."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype="float64"),
                          "risk": np.asarray(risk, dtype="float64")})
    frame["bin"] = pd.qcut(frame["risk"].rank(method="first"), q=n_bins,
                           labels=range(1, n_bins + 1))
    grouped = frame.groupby("bin", observed=True)
    return pd.DataFrame({
        "n": grouped.size(),
        "mean_predicted": grouped["risk"].mean(),
        "observed": grouped["y"].mean(),
    })


def compare_models(fitted: dict[str, FittedModel], test: pd.DataFrame,
                   target: str) -> pd.DataFrame:
    """Score every fitted model on the same test rows."""
    data = _labelled(test, target)
    y = data[target].astype("float64").to_numpy()
    rows = []
    for name, model in fitted.items():
        risk = model.risk(data)
        row = {"model": name, "label": model.spec.label, "family": model.spec.family,
               "citation": model.spec.citation, "n_train": model.n_train}
        row.update(score_metrics(y, risk))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("model")
    return frame.sort_values("average_precision", ascending=False)


def bootstrap_metric_difference(y_true: np.ndarray, risk_a: np.ndarray, risk_b: np.ndarray,
                                metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
                                n_boot: int = 500, random_state: int = RANDOM_STATE
                                ) -> dict[str, float]:
    """Paired bootstrap of ``metric(a) - metric(b)``.

    Resampling is paired — the same resampled rows score both models — so the
    interval reflects the *difference* rather than the sum of two independent
    sampling errors.  Returns the point estimate, a percentile 95% interval and
    the share of replicates in which ``a`` loses.

    Which metric matters depends on the question.  ROC-AUC asks whether the
    whole ranking improved; ``average_precision_score`` asks whether it improved
    *where the positives are*, which at a 1-2% base rate is usually the question
    worth asking.  The two can disagree, and when they do it is informative
    rather than contradictory.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    n = len(y_true)
    observed = metric(y_true, risk_a) - metric(y_true, risk_b)

    differences = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        if y_boot.min() == y_boot.max():
            differences[i] = np.nan
            continue
        differences[i] = metric(y_boot, risk_a[idx]) - metric(y_boot, risk_b[idx])
    valid = differences[np.isfinite(differences)]
    return {
        "difference": float(observed),
        "ci_low": float(np.percentile(valid, 2.5)),
        "ci_high": float(np.percentile(valid, 97.5)),
        "p_worse": float((valid <= 0).mean()),
        "n_boot": float(len(valid)),
    }


def bootstrap_auc_difference(y_true: np.ndarray, risk_a: np.ndarray, risk_b: np.ndarray,
                             n_boot: int = 500, random_state: int = RANDOM_STATE
                             ) -> dict[str, float]:
    """Paired bootstrap of ``AUC(a) - AUC(b)``; see
    :func:`bootstrap_metric_difference`.

    Kept as a named wrapper, with ``auc_difference`` first, because the pipeline
    writes these dictionaries straight to ``outputs/tables`` and the column order
    of a published artefact should not move when the implementation underneath is
    generalised.
    """
    out = bootstrap_metric_difference(y_true, risk_a, risk_b, metric=roc_auc_score,
                                      n_boot=n_boot, random_state=random_state)
    return {"auc_difference": out["difference"], "ci_low": out["ci_low"],
            "ci_high": out["ci_high"], "p_worse": out["p_worse"],
            "n_boot": out["n_boot"]}


def permutation_importance_table(model: FittedModel, test: pd.DataFrame, target: str,
                                 n_repeats: int = 5, top_n: int | None = None,
                                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Drop in average precision when each feature is shuffled.

    Permutation importance on the *test* year, so it measures what the model
    actually relies on out of sample rather than what it fitted in sample.
    Correlated features share credit and will each look weaker than they are;
    read the table as "what would I lose by losing this column", not as a
    causal decomposition.
    """
    rng = np.random.default_rng(random_state)
    data = _labelled(test, target).copy()
    y = data[target].astype("float64").to_numpy()
    baseline = average_precision_score(y, model.risk(data))

    rows = []
    for column in model.spec.columns:
        drops = []
        original = data[column].to_numpy(copy=True)
        for _ in range(n_repeats):
            data[column] = rng.permutation(original)
            drops.append(baseline - average_precision_score(y, model.risk(data)))
        data[column] = original
        rows.append({"feature": column, "ap_drop_mean": float(np.mean(drops)),
                     "ap_drop_std": float(np.std(drops))})

    table = pd.DataFrame(rows).sort_values("ap_drop_mean", ascending=False)
    table["baseline_average_precision"] = baseline
    return table.head(top_n) if top_n else table
