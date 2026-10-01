"""Predicting corporate failure in the Canadian charity sector from T3010 data.

The package is layered; each module depends only on the ones above it.

===================  ========================================================
:mod:`~charity_risk.config`      paths, the observation window, the temporal split
:mod:`~charity_risk.fields`      T3010 line codes, and which filer tier reports them
:mod:`~charity_risk.ingest`      raw CSV readers with a parquet cache
:mod:`~charity_risk.panel`       the charity-year panel: de-duplication, tier-aware
                                 blanks, filing presence
:mod:`~charity_risk.outcomes`    what counts as failure - permanent exit, and the
                                 Greenlee-Trussel net-asset decline
:mod:`~charity_risk.features`    ratios, from the Tuckman-Chang four outwards,
                                 plus the detailed-return (Schedule 6) group
:mod:`~charity_risk.dataset`     the modelling frame and its splits
:mod:`~charity_risk.benchmarks`  the classical models, reproduced
:mod:`~charity_risk.hs_index`    the HS risk index: five dimensions from the HS ratios
:mod:`~charity_risk.models`      pipelines and the specification registry
:mod:`~charity_risk.evaluate`    fitting, metrics, and paired comparison
:mod:`~charity_risk.hierarchical` a partially-pooled Bayesian alternative
:mod:`~charity_risk.plots`       the figure system
:mod:`~charity_risk.pipeline`    the end-to-end run
===================  ========================================================

Quick start::

    from charity_risk.dataset import load_dataset, exit_split
    from charity_risk.evaluate import fit_all, compare_models

    frame = load_dataset(years=[2020, 2021])
    train, test = exit_split(frame)
    results = compare_models(fit_all(train, "exit_next_year"), test, "exit_next_year")

The same, restricted to charities filing the detailed Schedule 6 return, with
the 46 extra features that return makes available::

    from charity_risk.dataset import load_dataset, schedule_6_split
    from charity_risk.models import SCHEDULE_6_SPECIFICATIONS

    frame = load_dataset(years=[2020, 2021])
    train, test = schedule_6_split(frame)
    results = compare_models(
        fit_all(train, "exit_next_year", specs=SCHEDULE_6_SPECIFICATIONS),
        test, "exit_next_year",
    )

That sample is *selected* — Schedule 6 is the form you file once you outgrow the
small-charity thresholds — so its metrics are not comparable with the
full-population ones above.  See ``notebooks/04-schedule-6-deep-dive``.

Or reproduce everything from the raw extract::

    python -m charity_risk.pipeline --refresh
"""

__version__ = "0.1.0"

__all__ = [
    "benchmarks",
    "config",
    "dataset",
    "evaluate",
    "features",
    "fields",
    "hs_index",
    "hierarchical",
    "ingest",
    "models",
    "outcomes",
    "panel",
    "pipeline",
    "plots",
]
