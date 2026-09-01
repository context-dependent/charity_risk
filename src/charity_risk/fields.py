"""T3010 line-code dictionary and filer-tier metadata.

The CRA public T3010 extract labels financial columns by the line number on the
form (``4200`` = total assets, and so on).  This module gives those codes names
and — more importantly — records *which* codes each tier of filer is actually
asked for.

Two filing tiers coexist in the same file, flagged by the
``Financial Indicator (D or 6)`` column:

``D``
    Section D of the T3010 itself.  A short-form statement available to small
    charities.  It collects totals only: total assets, total liabilities, total
    revenue, total expenditures and a coarse revenue breakdown.
``6``
    Schedule 6, the detailed statement.  It collects the full balance sheet and
    an itemised revenue and expenditure breakdown.

The distinction matters because a blank cell means two different things in the
two tiers.  For a Schedule 6 filer, a blank in line 4580 means "no investment
income" and should be read as zero.  For a Section D filer the same blank means
"never asked", and zero-filling it would manufacture a false ratio.  The
:data:`SECTION_D_COLLECTED` / :data:`SCHEDULE_6_ONLY` sets encode that, and
:func:`is_collected` is the single place the rest of the package consults.
"""

from __future__ import annotations

__all__ = [
    "FIELD_LABELS",
    "BALANCE_SHEET",
    "REVENUE_LINES",
    "EXPENSE_LINES",
    "SCHEDULE_6_ONLY",
    "SECTION_D_COLLECTED",
    "CORE_LINES",
    "REVENUE_SOURCES_CORE",
    "DESIGNATION_LABELS",
    "is_collected",
]

#: Line code -> short human label.  Descriptions are taken verbatim in spirit
#: from ``docs/T3010-Master-Public-Data-Dictionary-January-2021.xlsx``.
FIELD_LABELS: dict[str, str] = {
    # --- balance sheet: assets -------------------------------------------
    "4100": "cash_and_short_term_investments",
    "4110": "receivable_non_arms_length",
    "4120": "receivable_other",
    "4130": "investments_non_arms_length",
    "4140": "long_term_investments",
    "4150": "inventories",
    "4155": "land_and_buildings_canada",
    "4160": "other_capital_assets_canada",
    "4165": "capital_assets_outside_canada",
    "4166": "accumulated_amortization",
    "4170": "other_assets",
    "4180": "ten_year_gifts",
    "4200": "total_assets",
    "4250": "assets_not_used_in_charitable_activities",
    # --- balance sheet: liabilities --------------------------------------
    "4300": "accounts_payable",
    "4310": "deferred_revenue",
    "4320": "owing_non_arms_length",
    "4330": "other_liabilities",
    "4350": "total_liabilities",
    # --- revenue ----------------------------------------------------------
    "4500": "receipted_gifts",
    "4505": "ten_year_gifts_received",
    "4510": "gifts_from_other_charities",
    "4530": "other_gifts_not_receipted",
    "4540": "revenue_federal_government",
    "4550": "revenue_provincial_government",
    "4560": "revenue_municipal_government",
    "4570": "revenue_all_government",
    "4571": "receipted_revenue_outside_canada",
    "4575": "non_receipted_revenue_outside_canada",
    "4580": "interest_and_investment_income",
    "4590": "gross_proceeds_disposition",
    "4600": "net_proceeds_disposition",
    "4610": "rental_income",
    "4620": "membership_dues",
    "4630": "fundraising_revenue",
    "4640": "sale_of_goods_and_services",
    "4650": "other_revenue",
    "4700": "total_revenue",
    "5610": "receipted_tuition_fees",
    # --- expenditure ------------------------------------------------------
    "4800": "advertising_and_promotion",
    "4810": "travel_and_vehicle",
    "4820": "interest_and_bank_charges",
    "4830": "licences_memberships_dues",
    "4840": "office_supplies",
    "4850": "occupancy_costs",
    "4860": "professional_and_consulting_fees",
    "4870": "staff_education_and_training",
    "4880": "total_compensation",
    "4890": "donated_goods_used",
    "4891": "purchased_supplies_and_assets",
    "4900": "amortization",
    "4910": "research_grants_and_scholarships",
    "4920": "other_expenditures",
    "4950": "total_expenditures_before_donee_gifts",
    "5000": "expenditure_charitable_activities",
    "5010": "expenditure_management_and_admin",
    "5020": "expenditure_fundraising",
    "5030": "expenditure_political_activities",
    "5040": "expenditure_other",
    "5050": "gifts_to_qualified_donees",
    "5100": "total_expenditures",
    # --- disbursement quota ----------------------------------------------
    "5500": "amount_accumulated",
    "5510": "amount_disbursed",
    "5750": "special_reduction",
    "5900": "avg_property_not_charitable_24m",
    "5910": "avg_property_not_charitable_24m_alt",
}

BALANCE_SHEET: tuple[str, ...] = (
    "4100", "4110", "4120", "4130", "4140", "4150", "4155", "4160", "4165",
    "4166", "4170", "4180", "4200", "4250",
    "4300", "4310", "4320", "4330", "4350",
)

REVENUE_LINES: tuple[str, ...] = (
    "4500", "4505", "4510", "4530", "4540", "4550", "4560", "4570", "4571",
    "4575", "4580", "4590", "4600", "4610", "4620", "4630", "4640", "4650",
    "4700", "5610",
)

EXPENSE_LINES: tuple[str, ...] = (
    "4800", "4810", "4820", "4830", "4840", "4850", "4860", "4870", "4880",
    "4890", "4891", "4900", "4910", "4920", "4950", "5000", "5010", "5020",
    "5030", "5040", "5050", "5100",
)

#: Lines collected only on Schedule 6.  Empirically verified against the raw
#: extract: every one of these is 100% null for ``Financial Indicator == "D"``
#: in all five years (see ``notebooks/01-data-profile``).
SCHEDULE_6_ONLY: frozenset[str] = frozenset({
    "4100", "4110", "4120", "4130", "4140", "4150", "4155", "4160", "4165",
    "4166", "4170", "4180", "4250",
    "4300", "4310", "4320", "4330",
    "4505", "4540", "4550", "4560", "4571", "4580", "4590", "4600", "4610",
    "4620", "5610",
    "4800", "4810", "4820", "4830", "4840", "4850", "4860", "4870", "4880",
    "4890", "4891", "4900", "4910", "4920", "5020", "5030", "5040",
    "5500", "5510", "5750", "5900", "5910",
})

#: Lines the short Section D form does collect.
SECTION_D_COLLECTED: frozenset[str] = frozenset(FIELD_LABELS) - SCHEDULE_6_ONLY

#: Lines available for *every* filer, in every year, and therefore the only
#: ones a headline model may depend on without a tier interaction.
CORE_LINES: tuple[str, ...] = tuple(sorted(SECTION_D_COLLECTED))

#: Harmonised revenue decomposition used for the Tuckman-Chang concentration
#: index.  Section D filers report a coarser split than Schedule 6 filers, so
#: the Schedule 6 detail is collapsed into the same eight buckets.  The
#: ``"other"`` bucket absorbs investment income, rentals, membership dues and
#: net disposition proceeds, which a Section D filer would report on line 4650
#: anyway.  Values are the line codes summed into each bucket, keyed by tier.
REVENUE_SOURCES_CORE: dict[str, dict[str, tuple[str, ...]]] = {
    "receipted_gifts": {"D": ("4500",), "6": ("4500",)},
    "gifts_from_other_charities": {"D": ("4510",), "6": ("4510",)},
    "other_gifts": {"D": ("4530",), "6": ("4530",)},
    "government": {"D": ("4570",), "6": ("4540", "4550", "4560")},
    "foreign_non_receipted": {"D": ("4575",), "6": ("4575",)},
    "fundraising": {"D": ("4630",), "6": ("4630",)},
    "sales_of_goods_and_services": {"D": ("4640",), "6": ("4640",)},
    "other": {"D": ("4650",), "6": ("4650", "4580", "4600", "4610", "4620")},
}

DESIGNATION_LABELS: dict[str, str] = {
    "A": "public_foundation",
    "B": "private_foundation",
    "C": "charitable_organization",
}


def is_collected(line: str, tier: str) -> bool:
    """Return whether ``line`` is asked of a filer in ``tier`` (``"D"``/``"6"``).

    Unknown tiers are treated as Schedule 6, which is the conservative choice:
    it never converts a genuine blank into a spurious "not collected".
    """
    if tier == "D":
        return line in SECTION_D_COLLECTED
    return True
