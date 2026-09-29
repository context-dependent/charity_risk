"""Tests for the behaviours that are easy to get quietly wrong.

Not a coverage exercise.  These cover the four places where a plausible-looking
change would produce plausible-looking numbers that are wrong:

* currency parsing, because the CRA writes the minus sign outside the dollar sign;
* tier-aware blank handling, because zero and "never asked" look identical;
* the concentration index and safe ratios, because a zero denominator must not
  become a large finite number;
* the leakage guard, because a forward-looking column added to a feature group
  would improve every metric in the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from charity_risk.benchmarks import TC_RISK_DIRECTION, TuckmanChangScore
from charity_risk.dataset import schedule_6_split
from charity_risk.features import (
    FORBIDDEN_FEATURES,
    HS_RATIOS,
    SCHEDULE_6,
    add_schedule_6,
    assert_no_leakage,
    feature_columns,
    herfindahl,
    safe_ratio,
)
from charity_risk.ingest import parse_money
from charity_risk.models import (
    SCHEDULE_6_SPECIFICATIONS,
    SPECIFICATIONS,
    BlendEstimator,
    Winsorizer,
    build_elasticnet_pipeline,
    build_gbm_pipeline,
    build_mlp_pipeline,
)
from charity_risk.outcomes import add_exit_labels, labelled_years
from charity_risk.panel import apply_tier_missingness, dedupe_returns
from charity_risk.hs_index import (
    HS_DIMENSIONS,
    HS_RISK_DIRECTION,
    HSRiskIndex,
)


# --------------------------------------------------------------- parsing
def test_parse_money_handles_cra_formats():
    series = pd.Series(["$1234.00", "-$1321.00", "$0.00", None, "", "($50.00)"])
    parsed = parse_money(series)
    assert parsed.tolist()[:3] == [1234.0, -1321.0, 0.0]
    assert np.isnan(parsed[3]) and np.isnan(parsed[4])
    assert parsed[5] == -50.0


def test_parse_money_keeps_blanks_missing_not_zero():
    """A blank must survive parsing as NaN; only `panel` may decide it is nil."""
    assert parse_money(pd.Series([None])).isna().all()


# --------------------------------------------------------------- panel
def _tier_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "bn": ["A", "B"],
        "year": [2020, 2020],
        "filer_tier": pd.array(["6", "D"], dtype="string"),
        # 4200 (total assets) is collected on both forms.
        "4200": [100.0, np.nan],
        # 4100 (cash) is Schedule 6 only.
        "4100": [np.nan, np.nan],
    })


def test_blank_means_nil_for_schedule_6_and_missing_for_section_d():
    out = apply_tier_missingness(_tier_frame())
    # Schedule 6 filer left cash blank: it was asked, so the blank means zero.
    assert out.loc[0, "4100"] == 0.0
    # Section D filer was never asked for cash: the blank must stay missing.
    assert np.isnan(out.loc[1, "4100"])
    # Total assets is asked of both, so a blank is nil for both.
    assert out.loc[1, "4200"] == 0.0


def test_share_nil_lines_is_relative_to_what_the_tier_was_asked():
    out = apply_tier_missingness(_tier_frame())
    assert (out["share_nil_lines"] >= 0).all()
    assert (out["share_nil_lines"] <= 1).all()


def test_dedupe_keeps_latest_period_end_and_counts_returns():
    frame = pd.DataFrame({
        "bn": ["A", "A", "B"],
        "year": [2019, 2019, 2019],
        "fiscal_period_end": pd.to_datetime(["2019-05-31", "2019-12-31", "2019-03-31"]),
        "filer_tier": pd.array(["6", "6", "D"], dtype="string"),
        "4700": [684059.0, 421255.0, 1000.0],
    })
    out = dedupe_returns(frame).set_index("bn")
    assert out.loc["A", "4700"] == 421255.0          # latest period end wins
    assert out.loc["A", "n_returns_in_year"] == 2     # and the stub is flagged
    assert out.loc["B", "n_returns_in_year"] == 1


# --------------------------------------------------------------- ratios
def test_safe_ratio_returns_nan_for_vanishing_denominator():
    out = safe_ratio(pd.Series([100.0, 100.0]), pd.Series([50.0, 0.0]))
    assert out[0] == 2.0
    assert np.isnan(out[1])


def test_herfindahl_bounds_and_missing():
    components = pd.DataFrame({
        "a": [100.0, 25.0, 0.0],
        "b": [0.0, 25.0, 0.0],
        "c": [0.0, 25.0, 0.0],
        "d": [0.0, 25.0, 0.0],
    })
    out = herfindahl(components)
    assert out[0] == pytest.approx(1.0)     # single source
    assert out[1] == pytest.approx(0.25)    # four equal sources = 1/k
    assert np.isnan(out[2])                 # no revenue, no concentration


def test_herfindahl_clips_negative_components():
    components = pd.DataFrame({"a": [100.0], "b": [-40.0]})
    assert herfindahl(components)[0] == pytest.approx(1.0)


# --------------------------------------------------------------- outcomes
def test_exit_label_requires_lookahead():
    panel = pd.DataFrame({
        "bn": ["A", "B", "C"],
        "year": [2019, 2021, 2022],
        "last_filing_year": [2019, 2021, 2022],
    })
    out = add_exit_labels(panel, years=(2019, 2020, 2021, 2022, 2023), min_lookahead=2)
    assert out.loc[0, "exit_next_year"] == 1.0     # 4 lookahead years
    assert out.loc[1, "exit_next_year"] == 1.0     # exactly 2 lookahead years
    assert np.isnan(out.loc[2, "exit_next_year"])  # only 1 - not trustworthy
    assert out.loc[2, "exit_provisional"] == 1     # but recorded as provisional


def test_labelled_years_matches_the_window():
    assert labelled_years(panel_end=2023, years=(2019, 2020, 2021, 2022, 2023),
                          min_lookahead=2) == [2019, 2020, 2021]


# --------------------------------------------------------------- leakage
def test_no_specification_uses_a_forward_looking_column():
    for name, spec in SPECIFICATIONS.items():
        assert_no_leakage(spec.columns), name


def test_leakage_guard_actually_fires():
    with pytest.raises(ValueError, match="forward-looking"):
        assert_no_leakage(["log_total_revenue", "last_filing_year"])


def test_feature_groups_contain_no_forbidden_columns():
    every = feature_columns(["tuckman_chang", "trussel", "structure", "dynamics",
                             "composition", "liquidity", "filing", "categorical"])
    assert not set(every) & FORBIDDEN_FEATURES


# --------------------------------------------------------------- benchmarks
def test_winsorizer_leaves_rare_binary_flags_alone():
    """Clipping to the 1st-99th percentile would collapse a 2%-prevalence flag."""
    frame = pd.DataFrame({"flag": [0.0] * 98 + [1.0, 1.0],
                          "continuous": np.linspace(0, 1000, 100)})
    out = Winsorizer().fit_transform(frame)
    assert out[:, 0].max() == 1.0
    assert out[:, 1].max() < 1000.0


def test_winsorizer_neutralises_all_missing_columns():
    frame = pd.DataFrame({"never_observed": [np.nan] * 10,
                          "observed": np.arange(10, dtype="float64")})
    winsorizer = Winsorizer().fit(frame)
    out = winsorizer.transform(frame)
    assert winsorizer.empty_features_[0]
    assert (out[:, 0] == 0.0).all()


def test_tuckman_chang_flags_the_direction_the_paper_specifies():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({ratio: rng.normal(size=1000) for ratio in TC_RISK_DIRECTION})
    scorer = TuckmanChangScore().fit(frame)
    flags = scorer.flags(frame)

    for ratio, direction in TC_RISK_DIRECTION.items():
        flagged = frame.loc[flags[f"tc_flag_{ratio}"] == 1, ratio]
        clean = frame.loc[flags[f"tc_flag_{ratio}"] == 0, ratio]
        if direction < 0:
            assert flagged.max() <= clean.min()   # risky tail is the bottom
        else:
            assert flagged.min() >= clean.max()   # risky tail is the top


# --------------------------------------------------------------- schedule 6
def _schedule_6_frame() -> pd.DataFrame:
    """One Schedule 6 filer with a full balance sheet, one Section D filer."""
    lines = {
        "4100": [200.0, np.nan], "4110": [10.0, np.nan], "4120": [40.0, np.nan],
        "4130": [0.0, np.nan], "4140": [100.0, np.nan], "4150": [50.0, np.nan],
        "4155": [600.0, np.nan], "4160": [200.0, np.nan], "4165": [0.0, np.nan],
        "4166": [-200.0, np.nan], "4170": [0.0, np.nan], "4250": [0.0, np.nan],
        "4300": [100.0, np.nan], "4310": [0.0, np.nan], "4320": [0.0, np.nan],
        "4330": [0.0, np.nan],
        "4540": [30.0, np.nan], "4550": [0.0, np.nan], "4560": [0.0, np.nan],
        "4580": [5.0, np.nan], "4590": [100.0, np.nan], "4600": [25.0, np.nan],
        "4610": [0.0, np.nan], "4620": [0.0, np.nan],
        "4800": [10.0, np.nan], "4810": [10.0, np.nan], "4820": [20.0, np.nan],
        "4830": [0.0, np.nan], "4840": [10.0, np.nan], "4850": [0.0, np.nan],
        "4860": [0.0, np.nan], "4870": [0.0, np.nan], "4880": [50.0, np.nan],
        "4890": [0.0, np.nan], "4891": [0.0, np.nan], "4900": [0.0, np.nan],
        "4910": [0.0, np.nan], "4920": [0.0, np.nan],
        "5000": [700.0, np.nan], "5050": [0.0, np.nan],
        "5900": [10_000.0, np.nan], "5910": [8_000.0, np.nan],
    }
    frame = pd.DataFrame(lines)
    frame["total_assets"] = [1000.0, 500.0]
    frame["total_liabilities"] = [100.0, 50.0]
    frame["total_revenue"] = [900.0, 400.0]
    frame["total_expenditures"] = [800.0, 350.0]
    frame["expenditures_before_donee_gifts"] = [800.0, 350.0]
    return frame


def test_schedule_6_features_are_missing_for_section_d_filers():
    out = add_schedule_6(_schedule_6_frame())
    for name in SCHEDULE_6:
        if name.startswith("delta_"):
            continue           # deltas are produced later, in add_dynamics
        assert np.isnan(out.loc[1, name]), name


def test_capital_asset_depletion_uses_the_magnitude_of_the_contra_asset():
    """4166 is stored negative; the ratio must land in [0, 1], not go negative."""
    out = add_schedule_6(_schedule_6_frame())
    # |−200| / (600 + 200 + 0)
    assert out.loc[0, "capital_asset_depletion"] == pytest.approx(0.25)
    # net book value: (600 + 200 + 0 − 200) / 1000
    assert out.loc[0, "capital_assets_to_assets"] == pytest.approx(0.60)


def test_disbursement_quota_ratio_and_flag():
    out = add_schedule_6(_schedule_6_frame())
    # base = max(5900, 5910) = 10_000; quota = 3.5% = 350; spending = 5000 + 5050 = 700
    assert out.loc[0, "disbursement_quota_ratio"] == pytest.approx(2.0)
    assert out.loc[0, "below_disbursement_quota"] == 0.0
    assert np.isnan(out.loc[1, "disbursement_quota_ratio"])


def test_structural_zero_indicators_mark_known_zeros_not_unknowns():
    """A Schedule 6 filer reporting nothing on a line reported a zero.

    The ratio built on it is undefined, but the *fact* is data and must survive
    as an indicator.  A Section D filer, never asked, gets NaN for both.
    """
    frame = _schedule_6_frame()
    frame.loc[0, ["4300", "4310", "4320"]] = 0.0   # no current liabilities
    frame.loc[0, "4820"] = 0.0                      # no interest paid
    out = add_schedule_6(frame)

    assert out.loc[0, "no_current_liabilities"] == 1.0
    assert np.isnan(out.loc[0, "current_ratio"])    # denominator is a known zero
    assert out.loc[0, "no_interest_expense"] == 1.0
    assert np.isnan(out.loc[0, "interest_coverage"])
    assert out.loc[0, "no_capital_assets"] == 0.0   # it does have capital assets

    # Section D: the lines were never asked, so the indicator is unknown too.
    for name in ("no_current_liabilities", "no_capital_assets",
                 "no_interest_expense", "no_quota_base"):
        assert np.isnan(out.loc[1, name]), name


def test_interest_coverage_matches_its_definition():
    out = add_schedule_6(_schedule_6_frame())
    # (revenue − expenditure + interest) / interest = (900 − 800 + 20) / 20
    assert out.loc[0, "interest_coverage"] == pytest.approx(6.0)


def test_current_ratio_uses_current_items_only():
    out = add_schedule_6(_schedule_6_frame())
    # (4100 + 4110 + 4120 + 4150) / (4300 + 4310 + 4320) = 300 / 100
    assert out.loc[0, "current_ratio"] == pytest.approx(3.0)


def test_schedule_6_split_keeps_only_detailed_filers():
    frame = pd.DataFrame({
        "bn": ["A", "A", "A", "B", "B", "B"],
        "year": [2019, 2020, 2021, 2019, 2020, 2021],
        "filer_tier": pd.array(["6", "6", "6", "D", "D", "6"], dtype="string"),
        "exit_next_year": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    })
    train, test = schedule_6_split(frame, train_year=2020, test_year=2021)
    assert train["bn"].tolist() == ["A"]          # B filed Section D in 2020
    assert sorted(test["bn"]) == ["A", "B"]

    _, strict_test = schedule_6_split(frame, train_year=2020, test_year=2021,
                                      require_lagged_tier=True)
    # B is dropped from 2021 too: it was not a Schedule 6 filer in 2020.
    assert strict_test["bn"].tolist() == ["A"]


def test_schedule_6_split_rejects_a_missing_lag_year_instead_of_emptying():
    frame = pd.DataFrame({
        "bn": ["A", "A"],
        "year": [2020, 2021],
        "filer_tier": pd.array(["6", "6"], dtype="string"),
        "exit_next_year": [0.0, 1.0],
    })
    with pytest.raises(ValueError, match="preceding year"):
        schedule_6_split(frame, train_year=2020, test_year=2021, require_lagged_tier=True)


def test_schedule_6_specifications_are_leakage_free_and_hyperparameter_matched():
    for name, spec in SCHEDULE_6_SPECIFICATIONS.items():
        assert_no_leakage(spec.columns), name

    core = SCHEDULE_6_SPECIFICATIONS["gradient_boosting_core"].build()
    richer = SCHEDULE_6_SPECIFICATIONS["schedule_6_gbm"].build()
    keys = ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf")
    core_params = core.named_steps["clf"].get_params()
    richer_params = richer.named_steps["clf"].get_params()
    assert [core_params[k] for k in keys] == [richer_params[k] for k in keys]


def test_schedule_6_group_adds_no_columns_the_core_set_already_has():
    core = set(feature_columns([g for g in ("tuckman_chang", "trussel", "structure",
                                            "dynamics", "composition", "liquidity",
                                            "filing")]))
    assert not core & set(SCHEDULE_6)


def _toy_blend_frame(n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    numeric = pd.DataFrame({
        "revenue_growth": rng.normal(size=n),
        "operating_margin": rng.normal(size=n),
        "log_total_revenue": rng.normal(loc=10.0, scale=2.0, size=n),
    })
    y = (rng.random(n) < 0.2).astype(float)
    return numeric, y


def test_blend_estimator_predict_proba_is_the_mean_of_its_three_components():
    """The blend must be exactly the average of independently fitted components.

    Both fits use the same fixed ``RANDOM_STATE``, so refitting the three
    pipelines outside the blend and averaging by hand must reproduce
    ``BlendEstimator.predict_proba`` bit for bit -- this is the property that
    lets the docstring claim a plain average rather than something fancier.
    """
    numeric = ["revenue_growth", "operating_margin", "log_total_revenue"]
    X, y = _toy_blend_frame()

    blend = BlendEstimator(numeric).fit(X, y)
    blended = blend.predict_proba(X)[:, 1]

    manual = np.mean([
        build_elasticnet_pipeline(numeric).fit(X, y).predict_proba(X)[:, 1],
        build_gbm_pipeline(numeric).fit(X, y).predict_proba(X)[:, 1],
        build_mlp_pipeline(numeric).fit(X, y).predict_proba(X)[:, 1],
    ], axis=0)

    np.testing.assert_allclose(blended, manual)
    assert set(blend.estimators_) == {"logit", "gbm", "mlp"}


def test_tuckman_chang_does_not_count_missing_ratios_as_flags():
    frame = pd.DataFrame({ratio: [0.0, np.nan] for ratio in TC_RISK_DIRECTION})
    scorer = TuckmanChangScore().fit(pd.DataFrame(
        {ratio: [-1.0, 0.0, 1.0] for ratio in TC_RISK_DIRECTION}))
    assert scorer.n_ratios_scored(frame).tolist() == [4, 0]
    assert scorer.score(frame)[1] == 0.0


# --------------------------------------------------------------- risk index
def _hs_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({ratio: rng.normal(size=n) for ratio in HS_RISK_DIRECTION})


def test_hs_dimensions_partition_the_hs_ratios():
    members = [r for ratios in HS_DIMENSIONS.values() for r in ratios]
    assert sorted(members) == sorted(HS_RATIOS)
    assert len(members) == len(set(members))


def test_hs_index_percentile_is_oriented_so_one_is_riskiest():
    frame = _hs_frame()
    index = HSRiskIndex().fit(frame)
    scores = index.ratio_scores(frame)
    for ratio, direction in HS_RISK_DIRECTION.items():
        riskiest = frame[ratio].idxmin() if direction < 0 else frame[ratio].idxmax()
        assert scores.loc[riskiest, ratio] > 0.99
        assert scores[ratio].between(0, 1).all()


def test_hs_index_flags_match_the_tuckman_chang_benchmark_on_shared_ratios():
    frame = _hs_frame()
    index = HSRiskIndex(method="flags").fit(frame)
    scorer = TuckmanChangScore().fit(frame)
    flags = index.ratio_scores(frame)
    for ratio in TC_RISK_DIRECTION:
        np.testing.assert_array_equal(flags[ratio], scorer.flags(frame)[f"tc_flag_{ratio}"])


def test_hs_index_weights_dimensions_not_ratios():
    """Profitability has four ratios and efficiency one; each dimension counts once."""
    reference = _hs_frame()
    index = HSRiskIndex(method="flags").fit(reference)
    row = pd.DataFrame({ratio: [0.0] for ratio in HS_RISK_DIRECTION})
    for ratio in HS_DIMENSIONS["profitability"]:
        row[ratio] = -10.0                      # every profitability ratio flagged
    dims = index.dimension_scores(row).iloc[0]
    assert dims["profitability"] == 1.0
    assert index.index(row).iloc[0] == pytest.approx(1.0 / len(HS_DIMENSIONS))


def test_hs_index_skips_missing_dimensions_instead_of_scoring_them_safe():
    reference = _hs_frame()
    index = HSRiskIndex().fit(reference)
    row = pd.DataFrame({ratio: [-10.0] for ratio in HS_RISK_DIRECTION})
    row["revenue_growth_volatility"] = 10.0
    row["revenue_concentration"] = 10.0
    full = index.index(row).iloc[0]
    row[list(HS_DIMENSIONS["liquidity"])] = np.nan   # a Section D filer
    assert index.n_dimensions_scored(row).iloc[0] == len(HS_DIMENSIONS) - 1
    assert index.index(row).iloc[0] == pytest.approx(full, abs=0.01)
    assert np.isnan(index.dimension_flags(row).iloc[0]["liquidity"])


def test_hs_index_leaves_ratios_unseen_in_training_unscored():
    reference = _hs_frame()
    reference["revenue_growth_volatility"] = np.nan  # the 2020 training year
    index = HSRiskIndex().fit(reference)
    assert index.unscored_ratios_ == ["revenue_growth_volatility"]
    scores = index.ratio_scores(_hs_frame(seed=1))
    assert scores["revenue_growth_volatility"].isna().all()
    assert scores["revenue_concentration"].notna().all()


def test_hs_index_probability_is_monotone_in_the_index():
    frame = _hs_frame(n=2000)
    index = HSRiskIndex().fit(frame)
    raw = index.index(frame)
    y = (np.random.default_rng(1).random(len(frame)) < 0.05 + 0.3 * raw).astype(float)
    fitted = index.fit(frame, y)
    risk = fitted.predict_proba(frame)[:, 1]
    order = np.argsort(raw.to_numpy())
    assert (np.diff(risk[order]) >= -1e-12).all()


def test_hs_index_specifications_fit_through_the_registry():
    from charity_risk.evaluate import fit_spec

    frame = _hs_frame(n=1000)
    frame["exit_next_year"] = (np.random.default_rng(2).random(len(frame)) < 0.1).astype(float)
    for name in ("hs_index", "hs_index_flags"):
        model = fit_spec(SPECIFICATIONS[name], frame, "exit_next_year")
        risk = model.risk(frame)
        assert risk.shape == (len(frame),)
        assert np.isfinite(risk).all()
