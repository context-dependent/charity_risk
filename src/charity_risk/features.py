"""Feature construction, from the 1991 classics outwards.

The features are organised into named groups so that a model specification is
a list of group names rather than a hand-copied list of columns.  The groups
nest deliberately:

``TUCKMAN_CHANG``
    The four measures of Tuckman and Chang (1991): equity balance, revenue
    concentration, administrative cost ratio and operating margin.
``TRUSSEL``
    The Trussel (2002) specification: revenue concentration, surplus margin,
    debt ratio, administrative cost ratio, plus a size control and sector
    fixed effects.
``STRUCTURE`` / ``DYNAMICS`` / ``COMPOSITION`` / ``LIQUIDITY`` / ``FILING``
    Our extensions: organisational attributes, year-on-year growth, revenue and
    expenditure mix, balance-sheet liquidity, and filing behaviour.
``SCHEDULE_6``
    The detailed-return extension: itemised balance sheet, itemised cost
    structure, disposal activity, interest coverage and the disbursement quota.
    Every feature here is ``NaN`` for the third of filers who use the short
    Section D form, so this group is meaningful only on the restricted sample —
    see :func:`charity_risk.dataset.schedule_6_split` and
    ``notebooks/04-schedule-6-deep-dive``.

Two rules are enforced throughout:

**No look-ahead.**  Every feature for year *t* is computable from returns filed
for years *t* and earlier.  Panel bookkeeping columns that peek forward
(``last_filing_year``, ``survives_window``, ``has_filing_gap``, ``n_years_filed``)
are listed in :data:`FORBIDDEN_FEATURES` and asserted against in
:func:`assert_no_leakage`.

**Ratios do not invent denominators.**  A ratio whose denominator is zero or
near-zero is ``NaN``, never a large finite number.  Roughly 6% of charity-years
report zero revenue, and silently mapping those to a margin of -1 would put a
large, structurally distinct group in the tail of a continuous feature.

Note what that ``NaN`` does and does not mean.  For a filer that *was asked* the
question, a blank line is a reported **zero**, not an unknown — see
:func:`charity_risk.panel.apply_tier_missingness`.  So a ratio that comes out
``NaN`` here is usually undefined because its denominator is a known zero, not
because anything is missing.  That distinction is why the ``no_*`` indicators in
:data:`SCHEDULE_6` exist: the state is informative and deserves to be a feature,
not a hole.  Genuine missingness arises in exactly one place — the Schedule 6
lines that a Section D filer was never asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fields import FIELD_LABELS, REVENUE_SOURCES_CORE

__all__ = [
    "REQUIRED_PANEL_COLUMNS",
    "SCHEDULE_6",
    "SCHEDULE_6_DELTAS",
    "DISBURSEMENT_QUOTA_RATE",
    "add_schedule_6",
    "TUCKMAN_CHANG",
    "TRUSSEL_RATIOS",
    "TRUSSEL_CONTROLS",
    "STRUCTURE",
    "DYNAMICS",
    "COMPOSITION",
    "LIQUIDITY",
    "FILING",
    "CATEGORICAL",
    "FEATURE_GROUPS",
    "FORBIDDEN_FEATURES",
    "safe_ratio",
    "herfindahl",
    "add_tuckman_chang",
    "add_trussel",
    "add_composition",
    "add_liquidity",
    "add_dynamics",
    "add_structure",
    "build_features",
    "feature_columns",
    "assert_no_leakage",
]

#: Denominators below this (dollars) are treated as absent rather than small.
#: One dollar of revenue is not a meaningful base for a percentage.
_DENOMINATOR_FLOOR = 1.0

#: Panel columns the feature and outcome code actually reads.  The full panel
#: carries every T3010 line under two names; loading a projection of it keeps
#: the build inside a couple of gigabytes.  Listed explicitly rather than
#: derived, so that adding a feature that needs a new line is a deliberate,
#: reviewable edit.
REQUIRED_PANEL_COLUMNS: tuple[str, ...] = (
    # keys and identity
    "bn", "year", "legal_name", "fiscal_period_end", "filer_tier", "form_id",
    "designation", "charity_type", "category_desc", "province",
    "registration_date", "age_years", "first_filing_year", "last_filing_year",
    # headline aggregates
    "total_assets", "total_liabilities", "net_assets", "total_revenue",
    "total_expenditures", "expenditures_before_donee_gifts",
    # harmonised revenue buckets
    "rev_receipted_gifts", "rev_gifts_from_other_charities", "rev_other_gifts",
    "rev_government", "rev_foreign_non_receipted", "rev_fundraising",
    "rev_sales_of_goods_and_services", "rev_other", "rev_bucket_total",
    # individual lines used by specific ratios
    "4100", "4110", "4120", "4300", "4310", "4320", "4880",
    "5000", "5010", "5020", "5050",
    # Schedule 6 detail (all null for short-form filers)
    "4130", "4140", "4150", "4155", "4160", "4165", "4166", "4170", "4250",
    "4330", "4540", "4550", "4560", "4580", "4590", "4600", "4610", "4620",
    "4800", "4810", "4820", "4830", "4840", "4850", "4860", "4870", "4890",
    "4891", "4900", "4910", "4920", "5900", "5910",
    # filing behaviour
    "is_section_d", "is_accrual", "share_nil_lines", "n_returns_in_year",
    "fiscal_month",
)

TUCKMAN_CHANG: tuple[str, ...] = (
    "equity_balance",
    "revenue_concentration",
    "admin_cost_ratio",
    "operating_margin",
)

HS_RATIOS: tuple[str, ...] = (
    "equity_balance", 
    "net_assets_to_assets", 
    "equity_ratio", 
    "liquidity_ratio", 
    "working_capital_months", 
    "return_on_assets", 
    "net_revenue", 
    "operating_margin", 
    "operating_markup", 
    "admin_cost_ratio", 
    "revenue_growth_volatility", 
    "revenue_concentration"
)

TRUSSEL_RATIOS: tuple[str, ...] = (
    "revenue_concentration",
    "surplus_margin",
    "debt_ratio",
    "admin_cost_ratio",
)

TRUSSEL_CONTROLS: tuple[str, ...] = ("log_total_assets",)

STRUCTURE: tuple[str, ...] = (
    "log_total_revenue",
    "log_total_assets",
    "log_age",
    "is_new_entrant",
    "net_assets_to_assets",
    "liabilities_to_revenue",
    "expense_coverage",
    "negative_net_assets",
    "zero_revenue",
    "insolvent",
    "micro_charity",
)

DYNAMICS: tuple[str, ...] = (
    "revenue_growth",
    "asset_growth",
    "expenditure_growth",
    "net_asset_growth",
    "revenue_growth_volatility",
    "consecutive_deficit_years",
    "revenue_decline_over_30pct",
    "tier_downgrade",
)

COMPOSITION: tuple[str, ...] = (
    "share_receipted_gifts",
    "share_gifts_from_other_charities",
    "share_other_gifts",
    "share_government",
    "share_foreign_non_receipted",
    "share_fundraising",
    "share_sales_of_goods_and_services",
    "share_other",
    "donation_dependence",
    "earned_income_share",
    "program_expense_ratio",
    "donee_gift_share",
)

LIQUIDITY: tuple[str, ...] = (
    "months_of_cash",
    "quick_ratio",
    "payables_to_expenditure",
    "deferred_revenue_share",
    "compensation_ratio",
    "fundraising_expense_ratio",
    "fundraising_efficiency",
)

FILING: tuple[str, ...] = (
    "is_section_d",
    "is_accrual",
    "share_nil_lines",
    "n_returns_in_year",
    "fiscal_month",
)

CATEGORICAL: tuple[str, ...] = ("designation", "charity_type", "province")

#: Ratios whose year-on-year change is worth having.  Computed in
#: :func:`add_dynamics` from the ratios rather than from the raw lines, so that
#: the line-code columns can be released before the (memory-heavy) lag step.
SCHEDULE_6_DELTAS: tuple[str, ...] = (
    "cash_to_assets", "current_ratio", "compensation_ratio", "months_of_cash",
)

#: The disbursement quota rate applicable across this window.  A registered
#: charity must spend at least this share of the average value of its property
#: not used directly in charitable activities on charitable activities and gifts
#: to qualified donees; persistent failure is grounds for revocation.  The rate
#: rose to 5% on the portion of the base above $1M for fiscal periods beginning
#: on or after 1 January 2023, which affects almost none of a 2019-2023
#: fiscal-period-end window.
DISBURSEMENT_QUOTA_RATE: float = 0.035

SCHEDULE_6: tuple[str, ...] = (
    # balance sheet composition
    "cash_to_assets",
    "receivables_to_assets",
    "investments_to_assets",
    "capital_assets_to_assets",
    "inventories_to_assets",
    "other_assets_to_assets",
    "capital_asset_depletion",
    "non_charitable_asset_share",
    "related_party_asset_share",
    "payables_to_liabilities",
    "deferred_revenue_to_liabilities",
    "related_party_liability_share",
    "other_liabilities_share",
    # working capital and debt service
    "current_ratio",
    "working_capital_months",
    "interest_coverage",
    "interest_burden",
    # revenue detail
    "share_federal_revenue",
    "share_provincial_revenue",
    "share_municipal_revenue",
    "government_tier_concentration",
    "share_investment_income",
    "share_rental_income",
    "share_membership_dues",
    "investment_yield",
    "asset_disposal_intensity",
    "disposal_net_margin",
    # itemised cost structure
    "occupancy_ratio",
    "professional_fees_ratio",
    "travel_ratio",
    "advertising_ratio",
    "office_supplies_ratio",
    "amortization_ratio",
    "interest_expense_ratio",
    "staff_training_ratio",
    "donated_goods_ratio",
    "purchased_supplies_ratio",
    "other_expenditure_ratio",
    "expenditure_concentration",
    # disbursement quota
    "dq_base_to_assets",
    "disbursement_quota_ratio",
    "below_disbursement_quota",
    # structural zeros
    "no_current_liabilities",
    "no_capital_assets",
    "no_interest_expense",
    "no_quota_base",
) + tuple(f"delta_{name}" for name in SCHEDULE_6_DELTAS)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "tuckman_chang": TUCKMAN_CHANG,
    "trussel": TRUSSEL_RATIOS + TRUSSEL_CONTROLS,
    "structure": STRUCTURE,
    "dynamics": DYNAMICS,
    "composition": COMPOSITION,
    "liquidity": LIQUIDITY,
    "filing": FILING,
    "categorical": CATEGORICAL,
    "schedule_6": SCHEDULE_6,
    "hs_ratios": HS_RATIOS
}

#: Panel bookkeeping that encodes the future.  Never a feature.
FORBIDDEN_FEATURES: frozenset[str] = frozenset({
    "last_filing_year", "n_years_filed", "has_filing_gap", "survives_window",
    "exit_next_year", "exit_provisional", "vulnerable", "net_asset_change_3y",
    "years_observed_after", "vulnerable_horizon_observable",
    "filed_2019", "filed_2020", "filed_2021", "filed_2022", "filed_2023",
})


def safe_ratio(numerator: pd.Series, denominator: pd.Series,
               floor: float = _DENOMINATOR_FLOOR) -> pd.Series:
    """Element-wise ratio that returns ``NaN`` for a vanishing denominator."""
    num = pd.to_numeric(numerator, errors="coerce").astype("float64")
    den = pd.to_numeric(denominator, errors="coerce").astype("float64")
    valid = den.abs() >= floor
    return pd.Series(
        np.divide(num.to_numpy(), den.to_numpy(),
                  out=np.full(len(num), np.nan),
                  where=valid.to_numpy() & np.isfinite(num.to_numpy())),
        index=num.index,
    )


def herfindahl(components: pd.DataFrame) -> pd.Series:
    """Herfindahl concentration index over revenue components.

    Negative component values (a net loss on disposition, say) are clipped to
    zero before computing shares, so that the index stays in ``[1/k, 1]``.
    Rows whose components sum to less than :data:`_DENOMINATOR_FLOOR` return
    ``NaN`` — an organisation with no revenue has no revenue concentration.
    """
    positive = components.clip(lower=0.0)
    total = positive.sum(axis=1)
    shares = positive.div(total.where(total >= _DENOMINATOR_FLOOR), axis=0)
    return (shares ** 2).sum(axis=1).where(total >= _DENOMINATOR_FLOOR)


def _log1p_signed(series: pd.Series) -> pd.Series:
    """``log(1+x)`` extended to negatives as ``-log(1+|x|)``.

    Total assets and revenue are occasionally negative in the extract (net-of-
    liability reporting errors, or genuine investment losses).  Dropping the
    sign would misrepresent them; a signed log keeps the ordering monotone.
    """
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return pd.Series(np.sign(values) * np.log1p(values.abs()))


def add_tuckman_chang(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add the four Tuckman-Chang (1991) vulnerability measures.

    ============================  ==========================================
    Measure                       Definition here
    ============================  ==========================================
    Equity balance                net assets / total revenue (4200-4350)/4700
    Revenue concentration         Herfindahl index over the eight harmonised
                                  revenue buckets
    Administrative cost ratio     management and administration / total
                                  expenditures before gifts to donees,
                                  5010/4950
    Operating margin              (total revenue - total expenditures) /
                                  total revenue, (4700-5100)/4700
    ============================  ==========================================

    Tuckman and Chang read *low* equity balance, *high* concentration, *low*
    administrative cost ratio and *low* operating margin as risky; the
    administrative ratio is inverted relative to intuition because they treat
    administrative spending as discretionary slack that can be cut in a shock.
    The scoring rule that turns these into a 0-4 index lives in
    :mod:`charity_risk.benchmarks`, since it needs quintile cut-points fitted
    on a training sample.
    """
    panel = panel.copy() if copy else panel
    bucket_cols = [f"rev_{bucket}" for bucket in REVENUE_SOURCES_CORE]

    panel["equity_balance"] = safe_ratio(panel["net_assets"], panel["total_revenue"])
    panel["revenue_concentration"] = herfindahl(panel[bucket_cols])
    panel["admin_cost_ratio"] = safe_ratio(panel["5010"],
                                           panel["expenditures_before_donee_gifts"])
    panel["operating_margin"] = safe_ratio(
        panel["total_revenue"] - panel["total_expenditures"], panel["total_revenue"]
    )
    return panel


def add_trussel(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add the Trussel (2002) ratios that are not already Tuckman-Chang.

    Trussel's surplus margin is the same quantity as the Tuckman-Chang
    operating margin; it is aliased rather than recomputed so that the two
    specifications are demonstrably using the same series.
    """
    panel = panel.copy() if copy else panel
    panel["surplus_margin"] = panel["operating_margin"]
    panel["debt_ratio"] = safe_ratio(panel["total_liabilities"], panel["total_assets"])
    panel["log_total_assets"] = _log1p_signed(panel["total_assets"])
    return panel


def add_composition(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add revenue-mix and expenditure-mix shares.

    Shares are taken over the *non-negative* part of each revenue bucket and
    the corresponding total, matching the convention in :func:`herfindahl`.
    A handful of buckets are reported negative (net losses on disposition being
    the usual cause); leaving them signed produces shares outside ``[0, 1]``
    that are not mix information but reporting artefacts.
    """
    panel = panel.copy() if copy else panel
    bucket_cols = [f"rev_{bucket}" for bucket in REVENUE_SOURCES_CORE]
    positive = panel[bucket_cols].clip(lower=0.0)
    total = positive.sum(axis=1)

    for bucket in REVENUE_SOURCES_CORE:
        panel[f"share_{bucket}"] = safe_ratio(positive[f"rev_{bucket}"], total)

    panel["donation_dependence"] = safe_ratio(
        positive["rev_receipted_gifts"] + positive["rev_other_gifts"]
        + positive["rev_gifts_from_other_charities"], total
    )
    panel["earned_income_share"] = safe_ratio(
        positive["rev_sales_of_goods_and_services"] + positive["rev_fundraising"], total
    )
    panel["program_expense_ratio"] = safe_ratio(
        panel["5000"], panel["expenditures_before_donee_gifts"]
    )
    panel["donee_gift_share"] = safe_ratio(panel["5050"], panel["total_expenditures"])
    return panel


def add_liquidity(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add balance-sheet liquidity and cost-structure ratios.

    Every ratio here depends on a line collected only on Schedule 6, so it is
    ``NaN`` for the ~32% of charity-years filed on the short Section D form.
    That is a property of the data, not a defect: the tier is itself a feature
    (``is_section_d``), and the tree-based model in
    :mod:`charity_risk.models` handles the missingness natively while the
    logistic model gets an explicit missing-indicator.
    """
    panel = panel.copy() if copy else panel
    panel["months_of_cash"] = 12.0 * safe_ratio(panel["4100"], panel["total_expenditures"])
    panel["quick_ratio"] = safe_ratio(
        panel["4100"] + panel["4110"] + panel["4120"],
        panel["4300"] + panel["4310"] + panel["4320"],
    )
    panel["payables_to_expenditure"] = safe_ratio(panel["4300"], panel["total_expenditures"])
    panel["deferred_revenue_share"] = safe_ratio(panel["4310"], panel["total_revenue"])
    panel["compensation_ratio"] = safe_ratio(
        panel["4880"], panel["expenditures_before_donee_gifts"]
    )
    panel["fundraising_expense_ratio"] = safe_ratio(
        panel["5020"], panel["expenditures_before_donee_gifts"]
    )
    panel["fundraising_efficiency"] = safe_ratio(
        panel["5020"],
        panel["rev_receipted_gifts"] + panel["rev_other_gifts"] + panel["rev_fundraising"],
    )
    return panel


def add_schedule_6(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add the detailed-return features available only on Schedule 6.

    Every ratio here is ``NaN`` for a Section D filer, because its inputs were
    never collected.  That is not missingness to be imputed away; it is a
    different questionnaire.  Use these features on the restricted sample from
    :func:`charity_risk.dataset.schedule_6_split`.

    Four things are being bought with the extra detail that the totals cannot
    express:

    *Asset quality, not just asset size.*  Two charities with a million dollars
    of assets are not alike if one holds cash and the other holds a fully
    depreciated building.  ``capital_asset_depletion`` uses the accumulated
    amortisation line (4166, stored negative, so its magnitude is taken) against
    gross capital assets, giving a rough age of the physical plant.

    *Debt service.*  ``interest_coverage`` — operating surplus before interest
    over interest paid — is the classic corporate-failure ratio and has no
    analogue in the nonprofit vulnerability literature, which works from totals.

    *Cost structure, not just its level.*  The Tuckman-Chang administrative cost
    ratio collapses everything into one number.  Occupancy, compensation,
    professional fees and amortisation behave differently under stress: rent is
    contractual, consulting is not.

    *The disbursement quota.*  A registered charity must spend at least
    :data:`DISBURSEMENT_QUOTA_RATE` of the average value of its non-charitable
    property on charitable activities and gifts to qualified donees.  Failing it
    is grounds for revocation, so ``disbursement_quota_ratio`` is the one
    feature in this project that is close to a *statutory* failure condition
    rather than a financial symptom.  It is defined only for the ~18% of
    Schedule 6 filers whose property base clears the reporting threshold.
    """
    panel = panel.copy() if copy else panel
    assets = panel["total_assets"]
    liabilities = panel["total_liabilities"]
    revenue = panel["total_revenue"]
    expenditure = panel["expenditures_before_donee_gifts"]

    # --- balance sheet composition ---------------------------------------
    gross_capital = panel["4155"] + panel["4160"] + panel["4165"]
    panel["cash_to_assets"] = safe_ratio(panel["4100"], assets)
    panel["receivables_to_assets"] = safe_ratio(panel["4110"] + panel["4120"], assets)
    panel["investments_to_assets"] = safe_ratio(panel["4130"] + panel["4140"], assets)
    # 4166 is stored as a negative contra-asset, so the sum is net book value.
    panel["capital_assets_to_assets"] = safe_ratio(gross_capital + panel["4166"], assets)
    panel["inventories_to_assets"] = safe_ratio(panel["4150"], assets)
    panel["other_assets_to_assets"] = safe_ratio(panel["4170"], assets)
    panel["capital_asset_depletion"] = safe_ratio(panel["4166"].abs(), gross_capital)
    panel["non_charitable_asset_share"] = safe_ratio(panel["4250"], assets)
    panel["related_party_asset_share"] = safe_ratio(panel["4110"] + panel["4130"], assets)

    panel["payables_to_liabilities"] = safe_ratio(panel["4300"], liabilities)
    panel["deferred_revenue_to_liabilities"] = safe_ratio(panel["4310"], liabilities)
    panel["related_party_liability_share"] = safe_ratio(panel["4320"], liabilities)
    panel["other_liabilities_share"] = safe_ratio(panel["4330"], liabilities)

    # --- working capital and debt service --------------------------------
    current_assets = panel["4100"] + panel["4110"] + panel["4120"] + panel["4150"] + panel["4170"]
    current_liabilities = panel["4300"] + panel["4310"] + panel["4320"] + panel["4330"]
    panel["current_assets"] = current_assets
    panel["current_liabilities"] = current_liabilities
    panel["working_capital"] = current_assets - current_liabilities
    panel["current_ratio"] = safe_ratio(current_assets, current_liabilities)
    panel["working_capital_months"] = 12.0 * safe_ratio(
        panel["working_capital"],
        panel["total_expenditures"],
    )
    interest = panel["4820"]
    panel["interest"] = interest
    panel["interest_coverage"] = safe_ratio(
        revenue - panel["total_expenditures"] + interest, interest
    )
    panel["interest_burden"] = safe_ratio(interest, revenue)

    # --- revenue detail ---------------------------------------------------
    panel["share_federal_revenue"] = safe_ratio(panel["4540"], revenue)
    panel["share_provincial_revenue"] = safe_ratio(panel["4550"], revenue)
    panel["share_municipal_revenue"] = safe_ratio(panel["4560"], revenue)
    panel["government_tier_concentration"] = herfindahl(
        panel[["4540", "4550", "4560"]]
    )
    panel["share_investment_income"] = safe_ratio(panel["4580"], revenue)
    panel["share_rental_income"] = safe_ratio(panel["4610"], revenue)
    panel["share_membership_dues"] = safe_ratio(panel["4620"], revenue)
    panel["investment_yield"] = safe_ratio(panel["4580"], panel["4130"] + panel["4140"])
    # Selling assets at a scale that matters relative to the balance sheet is a
    # distress signal the totals hide: 4590 nets out of 4200 by year end.
    panel["asset_disposal_intensity"] = safe_ratio(panel["4590"], assets)
    panel["disposal_net_margin"] = safe_ratio(panel["4600"], panel["4590"])

    # --- itemised cost structure -----------------------------------------
    cost_lines = {
        "occupancy_ratio": "4850",
        "professional_fees_ratio": "4860",
        "travel_ratio": "4810",
        "advertising_ratio": "4800",
        "office_supplies_ratio": "4840",
        "amortization_ratio": "4900",
        "interest_expense_ratio": "4820",
        "staff_training_ratio": "4870",
        "donated_goods_ratio": "4890",
        "purchased_supplies_ratio": "4891",
        "other_expenditure_ratio": "4920",
    }
    for name, line in cost_lines.items():
        panel[name] = safe_ratio(panel[line], expenditure)

    itemised = ["4800", "4810", "4820", "4830", "4840", "4850", "4860", "4870",
                "4880", "4890", "4891", "4900", "4910", "4920"]
    panel["expenditure_concentration"] = herfindahl(panel[itemised])

    # --- disbursement quota ----------------------------------------------
    # Lines 5900 and 5910 carry the same concept over different averaging
    # windows; the public dictionary gives them identical descriptions and does
    # not distinguish them.  The larger is taken, which is the conservative
    # reading of a spending *obligation*.
    dq_base = panel[["5900", "5910"]].max(axis=1)
    charitable_spending = panel["5000"] + panel["5050"]
    panel["dq_base_to_assets"] = safe_ratio(dq_base, assets)
    quota = DISBURSEMENT_QUOTA_RATE * dq_base
    panel["disbursement_quota_ratio"] = safe_ratio(charitable_spending, quota)
    panel["below_disbursement_quota"] = (
        (panel["disbursement_quota_ratio"] < 1.0).astype("float64")
        .where(panel["disbursement_quota_ratio"].notna())
    )

    # --- structural zeros -------------------------------------------------
    # Where one of the ratios above is undefined, the denominator is a *known
    # zero*, not an unknown: a Schedule 6 filer that leaves line 4300 blank has
    # reported no accounts payable.  Division by it is undefined, so the ratio
    # is NaN — but the fact itself is data, and a strong signal.  A charity with
    # no payables, no capital assets and no borrowings has almost no balance
    # sheet left.  These indicators say that in the open, rather than leaving
    # the models to infer it from a hole.
    #
    # They stay NaN for Section D filers, where the underlying lines really are
    # unknown rather than nil.
    section_d = panel["4300"].isna()
    for name, total in {
        "no_current_liabilities": current_liabilities,
        "no_capital_assets": gross_capital,
        "no_interest_expense": interest,
        "no_quota_base": dq_base,
    }.items():
        panel[name] = (total.abs() < _DENOMINATOR_FLOOR).astype("float64").where(~section_d)
    return panel


def add_dynamics(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add year-on-year change features.

    All lags are taken within ``bn`` over the *observed* years, and are ``NaN``
    when the previous year is missing.  ``revenue_growth_volatility`` is an
    expanding (never rolling-forward) standard deviation, so a year-*t* value
    uses only growth rates realised up to *t*.
    """
    panel = panel.copy() if copy else panel
    panel = panel.sort_values(["bn", "year"], kind="mergesort")
    grouped = panel.groupby("bn", observed=True, sort=False)

    log_revenue = _log1p_signed(panel["total_revenue"])
    log_assets = _log1p_signed(panel["total_assets"])
    log_expenditure = _log1p_signed(panel["total_expenditures"])
    log_net_assets = _log1p_signed(panel["net_assets"])

    prev_year = grouped["year"].shift(1)
    contiguous = (panel["year"] - prev_year).eq(1)

    def _diff(series: pd.Series) -> pd.Series:
        shifted = series.groupby(panel["bn"].to_numpy()).shift(1)
        return (series - shifted).where(contiguous)

    panel["revenue_growth"] = _diff(log_revenue)
    panel["asset_growth"] = _diff(log_assets)
    panel["expenditure_growth"] = _diff(log_expenditure)
    panel["net_asset_growth"] = _diff(log_net_assets)

    growth = panel["revenue_growth"]
    panel["revenue_growth_volatility"] = (
        growth.groupby(panel["bn"].to_numpy())
        .transform(lambda s: s.expanding(min_periods=2).std())
    )
    panel["revenue_decline_over_30pct"] = (growth < np.log(0.70)).astype("float64").where(
        growth.notna()
    )

    deficit = (panel["operating_margin"] < 0).astype("float64").where(
        panel["operating_margin"].notna(), 0.0
    )
    panel["consecutive_deficit_years"] = (
        deficit.groupby(panel["bn"].to_numpy())
        .transform(lambda s: s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1))
    )

    # Moving from the detailed Schedule 6 to the short Section D form means the
    # charity has dropped below the small-charity thresholds: a contraction
    # signal that is invisible in any single ratio.
    prev_tier = grouped["is_section_d"].shift(1)
    panel["tier_downgrade"] = (
        (panel["is_section_d"] == 1) & (prev_tier == 0)
    ).astype("float64").where(prev_tier.notna())

    # Schedule 6 deltas are taken on the ratios rather than the raw lines, so
    # that the line-code columns can be released before this step.
    for name in SCHEDULE_6_DELTAS:
        if name in panel.columns:
            panel[f"delta_{name}"] = _diff(panel[name])
    return panel


def add_structure(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Add size, age and solvency-state features."""
    panel = panel.copy() if copy else panel
    panel["log_total_revenue"] = _log1p_signed(panel["total_revenue"])
    panel["log_age"] = np.log1p(panel["age_years"].clip(lower=0))
    panel["is_new_entrant"] = (
        (panel["first_filing_year"] == panel["year"]) & (panel["year"] > panel["year"].min())
    ).astype("float64")
    panel["net_assets_to_assets"] = safe_ratio(panel["net_assets"], panel["total_assets"])
    panel["liabilities_to_revenue"] = safe_ratio(panel["total_liabilities"],
                                                 panel["total_revenue"])
    panel["expense_coverage"] = safe_ratio(panel["total_revenue"], panel["total_expenditures"])
    panel["negative_net_assets"] = (panel["net_assets"] < 0).astype("float64")
    panel["zero_revenue"] = (panel["total_revenue"].fillna(0).abs() < 1).astype("float64")
    panel["insolvent"] = (panel["total_liabilities"] > panel["total_assets"]).astype("float64")
    panel["micro_charity"] = (panel["total_revenue"].fillna(0) < 10_000).astype("float64")
    panel["n_returns_in_year"] = panel["n_returns_in_year"].astype("float64")
    return panel

def add_hs_ratios(panel: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Add ratios from HS index concept.
    Some already added to the panel. 
    Full set of hs features is defined as global HS_RATIOS 
    1 Accounting Constructs
      1.1 Solvency Ratios
          1.1.1 net assets / revenue is panel["equity_balance"]
          1.1.2 net assets / total assets is panel["net_assets_to_assets"]
          1.1.3 net assets / total liabilities is panel["equity_ratio"] = panel["net_assets"] / panel["total_liabilities"]
      1.2 Liquidity
          1.2.1 Working capital / total assets is panel["liquidity_ratio"] = panel["working_capital"] / panel["total_assets"]
          1.2.2 12 x Unrestricted Net Assets / Total Expenses is panel["working_capital_months"]
      1.3 Profitability
          1.3.1 ROA (return on assets) is panel["return_on_assets"] = panel["net_revenue"] / panel["total_assets"]
          1.3.2 surplus is panel["net_revenue"] = panel["total_revenue"] - panel["total_expenditures"]
          1.3.3 margin ratio is panel["operating_margin"]
          1.3.4 markup is panel["operating_markup"] = panel["net_revenue"] / panel["total_expenditures"]
    2 Efficiency
      2.1 ACR (administrative cost ratio) is panel["admin_cost_ratio"]
    3 Revenue
      3.1 Revenue volatility is panel["revenue_growth_volatility"] = sd(revenue_growth[t-3:t-1])
      3.2 Funding source concentration is panel["revenue_concentration"] = herfindahl(panel[revenue_cols])
    """
    
    panel = panel.copy() if copy else panel
    panel["equity_ratio"] = safe_ratio(panel["net_assets"], panel["total_liabilities"]) 
    panel["liquidity_ratio"] = safe_ratio(panel["working_capital"], panel["total_assets"])
    panel["net_revenue"] = panel["total_revenue"] - panel["total_expenditures"]
    panel["return_on_assets"] = safe_ratio(panel["net_revenue"], panel["total_assets"])
    panel["operating_markup"] = safe_ratio(panel["net_revenue"], panel["total_expenditures"])
    return panel


def build_features(panel: pd.DataFrame, drop_line_codes: bool = True) -> pd.DataFrame:
    """Run the full feature pipeline in dependency order.

    Once every ratio has been computed, the raw line-code columns are dropped
    unless ``drop_line_codes=False``.  They are eighty-odd columns that nothing
    downstream reads, and carrying them through the lagging step in
    :func:`add_dynamics` — which sorts and copies the whole frame — roughly
    doubles the peak memory of the build.  Pass ``False`` when auditing a ratio
    against the line it came from.

    The frame is copied **once**, here, and the individual builders then write
    into it.  Each builder still defends itself by default when called on its
    own; letting all six copy in sequence is what pushes the build through a
    2 GB ceiling.
    """
    panel = panel.copy()
    panel = add_tuckman_chang(panel, copy=False)
    panel = add_trussel(panel, copy=False)
    panel = add_composition(panel, copy=False)
    panel = add_liquidity(panel, copy=False)
    panel = add_schedule_6(panel, copy=False)
    if drop_line_codes:
        panel = panel.drop(columns=[c for c in FIELD_LABELS if c in panel.columns])
    panel = add_structure(panel, copy=False)
    panel = add_dynamics(panel, copy=False)
    panel = add_hs_ratios(panel, copy=False)
    return panel


def feature_columns(groups: list[str] | tuple[str, ...]) -> list[str]:
    """Resolve a list of group names to a de-duplicated column list."""
    resolved: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {group!r}; "
                           f"choose from {sorted(FEATURE_GROUPS)}")
        for column in FEATURE_GROUPS[group]:
            if column not in resolved:
                resolved.append(column)
    return resolved


def assert_no_leakage(columns: list[str] | tuple[str, ...]) -> None:
    """Raise if any forward-looking column has crept into a feature list."""
    offenders = sorted(set(columns) & FORBIDDEN_FEATURES)
    if offenders:
        raise ValueError(f"forward-looking columns used as features: {offenders}")
