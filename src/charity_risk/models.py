"""Model specifications and scikit-learn pipelines.

A :class:`ModelSpec` names a set of features and a way to turn them into a
fitted probability model.  The registry :data:`SPECIFICATIONS` holds every
model this project compares, from the 1991 scoring rule to a forecast-average
ensemble of the flexible learners, so that :mod:`charity_risk.evaluate` can
loop over them without special cases.

Preprocessing choices worth stating:

*Winsorisation.*  T3010 financial ratios have genuinely unbounded tails — an
operating margin of -38,000 belongs to a charity that reported one dollar of
revenue and spent forty thousand.  Those rows are real and should not be
dropped, but under a linear model they would set the coefficients on their own.
Numeric features are therefore clipped to training-sample 1st/99th percentiles.
The cut-points are *learned in the pipeline*, so cross-validation and the
out-of-time test never see them.

*Missingness.*  Roughly a third of charity-years are short-form (Section D)
filers for whom the detailed balance sheet does not exist.  The linear models
get median imputation plus a missing-indicator column; the gradient-boosting
model routes missing values natively and gets neither.  Imputing a median cash
balance for a filer who was never asked would be an invention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .benchmarks import GREENLEE_TRUSSEL_2000, TRUSSEL_2002
from .config import RANDOM_STATE
from .features import CATEGORICAL, FEATURE_GROUPS, assert_no_leakage, feature_columns

__all__ = [
    "Winsorizer",
    "ModelSpec",
    "SPECIFICATIONS",
    "SCHEDULE_6_SPECIFICATIONS",
    "build_logit_pipeline",
    "build_elasticnet_pipeline",
    "build_gbm_pipeline",
    "build_mlp_pipeline",
    "BlendEstimator",
    "build_blend_pipeline",
    "extended_features",
    "schedule_6_features",
]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned at fit time.

    ``NaN`` passes through untouched so that a downstream imputer or a
    missing-aware estimator can decide what to do with it.

    Columns with at most ``min_unique`` distinct values are left alone.  Without
    that guard a rare binary flag is destroyed rather than tamed: a
    new-entrant indicator with a mean of 0.019 has a 99th percentile of zero, so
    clipping to the 1st-99th range collapses it to a constant.

    A column that is *entirely* missing in the training sample is replaced by
    zero and recorded in ``empty_features_``.  This is not a cosmetic guard: in
    a five-year window the first modellable year has only one year of history,
    so ``revenue_growth_volatility`` — which needs two growth observations —
    does not exist in 2020 even though it exists in 2021.  Neutralising it
    states plainly that the training year could not learn from it, rather than
    letting an all-NaN column reach the gradient-boosting binner, which fails
    on it.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99, min_unique: int = 10):
        self.lower = lower
        self.upper = upper
        self.min_unique = min_unique

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.lower_bounds_ = np.array(frame.quantile(self.lower), dtype="float64")
        self.upper_bounds_ = np.array(frame.quantile(self.upper), dtype="float64")

        n_unique = frame.nunique(dropna=True).to_numpy()
        skip = (n_unique <= self.min_unique)
        # A degenerate quantile pair (all-NaN column) would clip everything away.
        skip |= ~np.isfinite(self.lower_bounds_) | ~np.isfinite(self.upper_bounds_)
        self.lower_bounds_[skip] = -np.inf
        self.upper_bounds_[skip] = np.inf

        self.skipped_ = skip
        self.empty_features_ = frame.notna().sum().to_numpy() == 0
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(pd.DataFrame(X), dtype="float64")
        values = np.clip(values, self.lower_bounds_, self.upper_bounds_)
        if self.empty_features_.any():
            values[:, self.empty_features_] = 0.0
        return values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _numeric_branch(impute: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [("winsorize", Winsorizer())]
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True,
                                              keep_empty_features=True)))
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _preprocessor(numeric: Sequence[str], categorical: Sequence[str],
                  impute: bool, one_hot: bool) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = [
        ("num", _numeric_branch(impute), list(numeric)),
    ]
    if categorical:
        if one_hot:
            encoder = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=50, sparse_output=False)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                                     unknown_value=-1, encoded_missing_value=-1)
        transformers.append(("cat", encoder, list(categorical)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_logit_pipeline(numeric: Sequence[str], categorical: Sequence[str] = (),
                         C: float = 1e6) -> Pipeline:
    """Unpenalised (well, barely penalised) logistic regression.

    Used for the literature replications, where the point is to reproduce a
    published specification rather than to maximise out-of-sample accuracy.

    .. math::

        \\Pr(y_i = 1 \\mid x_i) = \\sigma(\\beta_0 + \\beta^\\top x_i),
        \\qquad \\sigma(z) = \\frac{1}{1 + e^{-z}}

    fitted by maximum likelihood.  In plain terms: each feature gets a weight,
    the weighted sum is squashed into a probability, and the weights are chosen
    to make the observed exits as likely as possible under that probability
    model.  ``C=1e6`` is scikit-learn's inverse-regularisation strength turned
    almost off, so the fitted :math:`\\beta` is close to the classical
    maximum-likelihood estimate the source papers report.
    """
    return Pipeline([
        ("prep", _preprocessor(numeric, categorical, impute=True, one_hot=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])


def build_elasticnet_pipeline(numeric: Sequence[str], categorical: Sequence[str] = (),
                              C: float = 0.1, l1_ratio: float = 0.5) -> Pipeline:
    """Elastic-net penalised logistic regression on the extended feature set.

    Same model as :func:`build_logit_pipeline`, fit instead by penalised
    maximum likelihood:

    .. math::

        \\min_\\beta\\; -\\ell(\\beta) + \\frac{1}{C}\\left[
            \\tfrac{1-\\rho}{2}\\lVert\\beta\\rVert_2^2 + \\rho\\lVert\\beta\\rVert_1
        \\right], \\qquad \\rho = \\texttt{l1\\_ratio}

    where :math:`\\ell` is the logit log-likelihood.  In plain terms: the fit
    is nudged to prefer smaller, sparser coefficients, which is what makes it
    usable on 51 correlated features instead of the 4-6 of the classical
    specifications — an unpenalised fit on that many correlated ratios would
    be unstable, inflating standard errors and occasionally flipping signs.
    """
    return Pipeline([
        ("prep", _preprocessor(numeric, categorical, impute=True, one_hot=True)),
        ("clf", LogisticRegression(solver="saga", C=C, l1_ratio=l1_ratio,
                                   max_iter=4000, random_state=RANDOM_STATE)),
    ])


def build_mlp_pipeline(numeric: Sequence[str], categorical: Sequence[str] = (),
                       hidden_layer_sizes: tuple[int, ...] = (256, 128, 64),
                       alpha: float = 1e-2, **kwargs) -> Pipeline:
    """A feedforward deep network, after Alam, Gao and Jones (2021).

    .. math::

        \\hat p_i = \\sigma\\bigl(W_L\\,\\phi(W_{L-1}\\,\\phi(\\cdots
                    \\phi(W_1 x_i + b_1)\\cdots) + b_{L-1}) + b_L\\bigr)

    with :math:`\\phi` the ReLU nonlinearity, three hidden layers
    (:math:`L=4`) of widths 256, 128, 64, and every weight matrix
    :math:`W_\\ell` fitted jointly by minimising binomial cross-entropy with
    Adam.  In plain terms: the same logistic-regression idea, but the
    weighted sum runs through several rounds of learned, nonlinear
    recombination before it is turned into a probability, so the model can
    represent interactions between ratios that a single linear combination
    cannot.

    Alam et al. find that a deep model beats a discrete-time hazard model at
    *identifying failures* (sensitivity 93.7% against 87.0%) on a panel of
    641,667 firm-months.  This is the corresponding rung on our ladder.

    Two things about the transplant are worth being explicit about.

    *The architecture is theirs in spirit, not in form.*  Their network is a
    GrNet operating on the Grassmann manifold, which needs a Riemannian
    autodiff framework; this is a plain multilayer perceptron trained with
    Adam.  What carries over is the depth-with-narrowing shape — their blocks
    run 80 -> 40 -> 20 — and the preprocessing discipline the paper insists on:
    winsorise at the 1st/99th percentile and normalise feature by feature,
    because neural networks, unlike gradient boosting, are badly behaved on raw
    financial ratios with unbounded tails.  Both are already in
    :func:`_preprocessor`.

    *The defaults were chosen by cross-validation on the training year alone*
    (3-fold, average precision), never on the test year.  The sweep is
    reproduced in ``notebooks/03-model-results``; every architecture tried
    landed within one standard deviation of every other, so the specific choice
    matters much less than the model class.

    No class re-weighting or resampling is applied.  Naive oversampling of the
    minority class was tried and is markedly *worse* — it distorts the
    probability scale the ranking metrics depend on.
    """
    params = dict(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        solver="adam",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    return Pipeline([
        ("prep", _preprocessor(numeric, categorical, impute=True, one_hot=True)),
        ("clf", MLPClassifier(**params)),
    ])


def build_gbm_pipeline(numeric: Sequence[str], categorical: Sequence[str] = (),
                       **kwargs) -> Pipeline:
    """Histogram gradient boosting with native missing-value handling.

    .. math::

        F_M(x) = \\sum_{m=1}^{M} \\nu\\, h_m(x), \\qquad
        h_m = \\arg\\min_h \\sum_i \\ell\\bigl(y_i,\\, F_{m-1}(x_i) + h(x_i)\\bigr)

    an additive expansion of shallow regression trees :math:`h_m` (at most
    ``max_leaf_nodes`` leaves each), fitted one at a time — each new tree
    targets the current model's residual error — with each tree's
    contribution shrunk by the learning rate :math:`\\nu`.  In plain terms:
    a sequence of simple decision rules, each one correcting what the rules
    before it got wrong, added together into a score.  Every split is a
    threshold test on one feature, so unlike the two logits above the model
    can represent a breakpoint (e.g. "risk jumps once interest coverage falls
    below 1x") without being told where to look for one.
    """
    params = dict(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    n_numeric = len(numeric)
    categorical_mask = [False] * n_numeric + [True] * len(categorical)
    return Pipeline([
        ("prep", _preprocessor(numeric, categorical, impute=False, one_hot=False)),
        ("clf", HistGradientBoostingClassifier(categorical_features=categorical_mask, **params)),
    ])


class BlendEstimator(BaseEstimator, ClassifierMixin):
    """Equal-weight average of the elastic-net logit, gradient boosting and MLP.

    A forecast-combination baseline, in the sense of Bates and Granger (1969):
    rather than choosing the single best-ranked predictor, average several that
    make different kinds of errors.  The three components here are chosen
    because :mod:`charity_risk.evaluate` shows them disagreeing in exactly the
    way combination is meant to exploit — different average-precision/AUC
    trade-offs, different calibration slopes (0.80, 0.89, 0.96 for gradient
    boosting, the MLP and the logit respectively in the 2021 test year) — and
    (calibration slopes of 0.80, 0.89 and 0.88 for gradient boosting, the MLP
    and the elastic-net logit respectively in the 2021 test year) — and because
    a plain average needs no held-out fold to fit safely, so it cannot leak the
    test year the way a stacked meta-learner fitted on in-sample base
    predictions could.

    .. math::

        \\hat p_i = \\tfrac{1}{3}\\left(\\hat p_i^{\\text{logit}}
                                        + \\hat p_i^{\\text{gbm}}
                                        + \\hat p_i^{\\text{mlp}}\\right)

    Each component is refitted here from its own pipeline factory rather than
    reusing an already-fitted estimator from a sibling specification.  That
    costs one extra fit of each of the three models, but it keeps every entry
    in :data:`SPECIFICATIONS` a self-contained, independently reproducible
    unit — consistent with how every other specification in this module is
    built — rather than threading a fitting order through
    :mod:`charity_risk.evaluate`.
    """

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str] = ()):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        numeric, categorical = list(self.numeric), list(self.categorical)
        self.estimators_ = {
            "logit": build_elasticnet_pipeline(numeric, categorical).fit(X, y),
            "gbm": build_gbm_pipeline(numeric, categorical).fit(X, y),
            "mlp": build_mlp_pipeline(numeric, categorical).fit(X, y),
        }
        return self

    def predict_proba(self, X):
        positive = np.mean(
            [estimator.predict_proba(X)[:, 1] for estimator in self.estimators_.values()],
            axis=0,
        )
        return np.column_stack([1.0 - positive, positive])


def build_blend_pipeline(numeric: Sequence[str], categorical: Sequence[str] = (),
                         **kwargs) -> BlendEstimator:
    """A forecast-average of the elastic-net logit, gradient boosting and MLP.

    See :class:`BlendEstimator`.  Kept as a named factory, matching every other
    ``build_*_pipeline`` function, so it slots into :func:`_extended_spec`
    without a special case.
    """
    return BlendEstimator(numeric, categorical)


#: Groups that apply to every filer.  ``schedule_6`` is excluded because it is
#: entirely missing for a third of the population; it is opted into explicitly
#: by :func:`schedule_6_features`.
_CORE_GROUPS = tuple(g for g in FEATURE_GROUPS if g not in ("categorical", "schedule_6"))


def _resolve(groups: tuple[str, ...]) -> tuple[list[str], list[str]]:
    numeric = [c for c in feature_columns(list(groups)) if c != "surplus_margin"]
    categorical = list(CATEGORICAL)
    assert_no_leakage(numeric + categorical)
    return numeric, categorical


def extended_features() -> tuple[list[str], list[str]]:
    """The full population feature set: ``(numeric, categorical)``.

    Everything in :data:`charity_risk.features.FEATURE_GROUPS` except the
    categorical group, which is returned separately, and the ``schedule_6``
    group, which does not exist for short-form filers.  ``surplus_margin`` is
    dropped as an exact alias of ``operating_margin``.
    """
    return _resolve(_CORE_GROUPS)


def schedule_6_features() -> tuple[list[str], list[str]]:
    """The extended set plus the detailed-return group.

    Only meaningful on the restricted sample from
    :func:`charity_risk.dataset.schedule_6_split`.
    """
    return _resolve(_CORE_GROUPS + ("schedule_6",))


@dataclass(frozen=True)
class ModelSpec:
    """A named, reproducible model: which columns, and how to fit them."""

    name: str
    label: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    factory: Callable[..., object]
    family: str = "fitted"
    citation: str = ""

    def build(self) -> object:
        return self.factory(self.numeric, self.categorical)

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)


def _extended_spec(name: str, label: str, factory: Callable[..., object],
                   citation: str = "", schedule_6: bool = False) -> ModelSpec:
    numeric, categorical = schedule_6_features() if schedule_6 else extended_features()
    return ModelSpec(name=name, label=label, numeric=tuple(numeric),
                     categorical=tuple(categorical), factory=factory, citation=citation)


SPECIFICATIONS: dict[str, ModelSpec] = {
    "size_only": ModelSpec(
        name="size_only",
        label="Size only (log revenue)",
        numeric=("log_total_revenue",),
        categorical=(),
        factory=build_logit_pipeline,
        family="baseline",
        citation="Null benchmark: how much of the classics is just organisational size?",
    ),
    "tuckman_chang_1991": ModelSpec(
        name="tuckman_chang_1991",
        label="Tuckman-Chang score (0-4)",
        numeric=tuple(GREENLEE_TRUSSEL_2000["numeric"]),
        categorical=(),
        factory=lambda numeric, categorical: None,  # handled by evaluate.fit_spec
        family="rule",
        citation="Tuckman & Chang (1991)",
    ),
    "greenlee_trussel_2000": ModelSpec(
        name="greenlee_trussel_2000",
        label="Greenlee-Trussel logit (4 TC ratios)",
        numeric=tuple(GREENLEE_TRUSSEL_2000["numeric"]),
        categorical=tuple(GREENLEE_TRUSSEL_2000["categorical"]),
        factory=build_logit_pipeline,
        citation="Greenlee & Trussel (2000)",
    ),
    "trussel_2002": ModelSpec(
        name="trussel_2002",
        label="Trussel logit (ratios + size + sector)",
        numeric=tuple(TRUSSEL_2002["numeric"]),
        categorical=tuple(TRUSSEL_2002["categorical"]),
        factory=build_logit_pipeline,
        citation="Trussel (2002)",
    ),
    "extended_logit": _extended_spec(
        "extended_logit", "Elastic-net logit (extended)", build_elasticnet_pipeline,
        citation="This project",
    ),
    "gradient_boosting": _extended_spec(
        "gradient_boosting", "Gradient boosting (extended)", build_gbm_pipeline,
        citation="This project",
    ),
    "deep_neural_net": _extended_spec(
        "deep_neural_net", "Deep neural net (extended)", build_mlp_pipeline,
        citation="Alam, Gao & Jones (2021)",
    ),
    "ensemble_blend": _extended_spec(
        "ensemble_blend", "Model-average ensemble (logit + GBM + net)", build_blend_pipeline,
        citation="Bates & Granger (1969); this project",
    ),
}

#: Specifications for the Schedule-6-only sample.  Deliberately a *separate*
#: registry rather than an addition to :data:`SPECIFICATIONS`: the main pipeline
#: runs on the whole population, where these features are missing for a third of
#: rows, and quietly widening the headline models would make the two sets of
#: results incomparable.
#:
#: The registry is built as a ladder, so the notebook can attribute any gain to
#: the thing that produced it: the classical benchmark, then the same extended
#: features used on the full population, then those plus the detailed return,
#: and finally the detailed return more or less on its own.
#:
#: The two gradient-boosting rungs share :data:`_SCHEDULE_6_GBM_PARAMS`, which is
#: a deeper configuration than the full-population default.  Matching them is
#: the point: with different hyperparameters the comparison would confound the
#: feature set with the amount of flexibility each model was granted.  The
#: settings were chosen by the sweep reproduced in
#: ``notebooks/04-schedule-6-deep-dive`` and favour neither rung.
_SCHEDULE_6_GBM_PARAMS = dict(max_leaf_nodes=63, min_samples_leaf=50,
                              learning_rate=0.03, max_iter=900)


def _matched_gbm(numeric, categorical):
    return build_gbm_pipeline(numeric, categorical, **_SCHEDULE_6_GBM_PARAMS)


def _schedule_6_only_features() -> tuple[list[str], list[str]]:
    """The detailed-return group plus two size controls, and nothing else."""
    numeric = list(FEATURE_GROUPS["schedule_6"]) + ["log_total_assets", "log_total_revenue"]
    assert_no_leakage(numeric)
    return numeric, []


SCHEDULE_6_SPECIFICATIONS: dict[str, ModelSpec] = {
    "trussel_2002": SPECIFICATIONS["trussel_2002"],
    "extended_logit_core": ModelSpec(
        name="extended_logit_core", label="Elastic-net logit (core)",
        numeric=SPECIFICATIONS["extended_logit"].numeric,
        categorical=SPECIFICATIONS["extended_logit"].categorical,
        factory=build_elasticnet_pipeline, citation="This project",
    ),
    "schedule_6_logit": _extended_spec(
        "schedule_6_logit", "Elastic-net logit (core + Schedule 6)",
        build_elasticnet_pipeline, citation="This project", schedule_6=True,
    ),
    "gradient_boosting_core": ModelSpec(
        name="gradient_boosting_core", label="Gradient boosting (core)",
        numeric=SPECIFICATIONS["gradient_boosting"].numeric,
        categorical=SPECIFICATIONS["gradient_boosting"].categorical,
        factory=_matched_gbm, citation="This project",
    ),
    "schedule_6_gbm": _extended_spec(
        "schedule_6_gbm", "Gradient boosting (core + Schedule 6)",
        _matched_gbm, citation="This project", schedule_6=True,
    ),
    "schedule_6_only_gbm": ModelSpec(
        name="schedule_6_only_gbm", label="Gradient boosting (Schedule 6 + size only)",
        numeric=tuple(_schedule_6_only_features()[0]), categorical=(),
        factory=_matched_gbm, citation="This project",
    ),
    # A third model family on the same two rungs, to test whether the deep net
    # behaves like the linear model (gains from the detailed return) or like the
    # tree (does not).
    "deep_neural_net_core": ModelSpec(
        name="deep_neural_net_core", label="Deep neural net (core)",
        numeric=SPECIFICATIONS["deep_neural_net"].numeric,
        categorical=SPECIFICATIONS["deep_neural_net"].categorical,
        factory=build_mlp_pipeline, citation="Alam, Gao & Jones (2021)",
    ),
    "schedule_6_mlp": _extended_spec(
        "schedule_6_mlp", "Deep neural net (core + Schedule 6)",
        build_mlp_pipeline, citation="Alam, Gao & Jones (2021)", schedule_6=True,
    ),
}
