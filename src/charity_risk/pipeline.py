"""End-to-end run: build, fit, evaluate, and write every artefact.

``python -m charity_risk.pipeline`` reproduces the whole project from the raw
CRA extract.  Everything the notebooks display is written here to
``outputs/tables`` (CSV) and ``outputs/figures`` (PNG), so the notebooks read
artefacts rather than re-fitting, and a reviewer can diff a run without opening
Jupyter.

Two prediction tasks are run:

``exit``
    Permanent deregistration one year ahead.  Fitted on 2020, tested on 2021.
``vulnerability``
    The Greenlee-Trussel three-year net-asset decline.  Fitted on 2019 (resolved
    2022), tested on 2020 (resolved 2023).  Included so that the benchmark
    models are also judged on the outcome they were designed for.

The 2022 cohort is then scored with the best exit model to produce a
forward-looking watchlist.  That cohort has no trustworthy label — which is the
point: it is the population a supervisor would actually be triaging.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .benchmarks import (  # noqa: E402
    GREENLEE_TRUSSEL_2000,
    TRUSSEL_2002,
    fit_statsmodels_logit,
)
from .config import (  # noqa: E402
    SCORE_YEAR,
    TABLE_DIR,
    TEST_YEAR,
    TRAIN_YEAR,
    ensure_dirs,
)
from .dataset import (  # noqa: E402
    build_dataset,
    exit_split,
    load_dataset,
    scoring_frame,
    vulnerability_split,
)
from .evaluate import (  # noqa: E402
    bootstrap_auc_difference,
    calibration_table,
    compare_models,
    fit_all,
    lift_table,
    permutation_importance_table,
)
from .models import SPECIFICATIONS  # noqa: E402
from .outcomes import outcome_summary  # noqa: E402
from . import plots  # noqa: E402

__all__ = ["TaskResult", "run_task", "run_watchlist", "run", "main"]

log = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Everything one prediction task produced."""

    name: str
    fitted: dict
    results: pd.DataFrame
    test: pd.DataFrame
    target: str
    risks: dict[str, "pd.Series"]


def _model_columns() -> list[str]:
    return sorted({column for spec in SPECIFICATIONS.values() for column in spec.columns})


def _write(frame: pd.DataFrame, name: str, index: bool = True) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / f"{name}.csv", index=index)
    log.info("wrote outputs/tables/%s.csv (%d rows)", name, len(frame))


def run_task(name: str, target: str, train: pd.DataFrame, test: pd.DataFrame,
             reference: str = "trussel_2002") -> TaskResult:
    """Fit every specification, score it, and write the comparison artefacts."""
    log.info("[%s] fitting %d specifications on %d rows (base rate %.3f%%)",
             name, len(SPECIFICATIONS), len(train), 100 * train[target].mean())
    fitted = fit_all(train, target)
    results = compare_models(fitted, test, target)
    _write(results, f"{name}_model_comparison")

    labelled = test.loc[test[target].notna()]
    y = labelled[target].to_numpy()
    risks = {model.spec.label: model.risk(labelled) for model in fitted.values()}

    # Paired bootstrap of every model against the strongest classical benchmark.
    reference_risk = fitted[reference].risk(labelled)
    rows = []
    for key, model in fitted.items():
        if key == reference:
            continue
        stats = bootstrap_auc_difference(y, model.risk(labelled), reference_risk)
        rows.append({"model": key, "label": model.spec.label,
                     "versus": fitted[reference].spec.label, **stats})
    _write(pd.DataFrame(rows).set_index("model"), f"{name}_auc_vs_benchmark")

    best = results.index[0]
    lift = lift_table(y, fitted[best].risk(labelled))
    _write(lift, f"{name}_lift_deciles_{best}")

    calibrations = {
        fitted[key].spec.label: calibration_table(y, fitted[key].risk(labelled))
        for key in results.index[:4]
    }

    # --- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    plots.plot_roc_curves(y, risks, ax=ax)
    plots.save_figure(fig, f"{name}_roc")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    plots.plot_pr_curves(y, risks, ax=ax)
    plots.save_figure(fig, f"{name}_precision_recall")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    plots.plot_gains_curves(y, risks, ax=ax)
    plots.save_figure(fig, f"{name}_gains")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    plots.plot_calibration(calibrations, ax=ax)
    plots.save_figure(fig, f"{name}_calibration")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plots.plot_decile_lift(lift, ax=ax,
                           title=f"Lift by risk decile - {fitted[best].spec.label}")
    plots.save_figure(fig, f"{name}_lift_deciles")
    plt.close(fig)

    for metric, fmt in [("average_precision", "{:.3f}"), ("roc_auc", "{:.3f}"),
                        ("lift_top5pct", "{:.1f}x")]:
        fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(results) + 1.8))
        plots.plot_metric_bars(results, metric=metric, ax=ax, label_format=fmt)
        plots.save_figure(fig, f"{name}_{metric}")
        plt.close(fig)

    importance = permutation_importance_table(fitted[best], test, target, n_repeats=5)
    _write(importance, f"{name}_permutation_importance", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    plots.plot_permutation_importance(importance, ax=ax)
    plots.save_figure(fig, f"{name}_permutation_importance")
    plt.close(fig)

    return TaskResult(name=name, fitted=fitted, results=results, test=labelled,
                      target=target, risks=risks)


def run_watchlist(exit_task: TaskResult, frame: pd.DataFrame, year: int = SCORE_YEAR,
                  top_n: int = 500) -> pd.DataFrame:
    """Score the most recent cohort and write the highest-risk organisations.

    The output is a triage list, not a verdict.  A high score says the
    organisation resembles those that deregistered a year later; it does not say
    the organisation is failing, and the exit label itself includes solvent
    wind-ups and amalgamations.
    """
    cohort = scoring_frame(frame, year=year)
    best = exit_task.results.index[0]
    model = exit_task.fitted[best]
    cohort = cohort.assign(risk=model.risk(cohort))

    columns = ["bn", "legal_name", "province", "charity_type", "designation",
               "total_revenue", "total_assets", "net_assets", "operating_margin",
               "months_of_cash", "consecutive_deficit_years", "revenue_growth",
               "is_section_d", "risk"]
    columns = [column for column in columns if column in cohort.columns]
    watchlist = cohort.sort_values("risk", ascending=False)[columns].head(top_n)
    watchlist.insert(0, "rank", range(1, len(watchlist) + 1))
    _write(watchlist, f"watchlist_{year}", index=False)
    return watchlist


def run(refresh: bool = False, skip_vulnerability: bool = False) -> dict[str, TaskResult]:
    """Run the whole project and return the task results."""
    ensure_dirs()
    plots.use_project_style()
    started = time.time()

    if refresh:
        log.info("rebuilding the analysis dataset from the raw extract")
        build_dataset(refresh=True)

    columns = _model_columns() + [
        "exit_next_year", "exit_provisional", "vulnerable", "legal_name",
        "total_revenue", "total_assets", "net_assets", "category_desc",
        "last_filing_year",
    ]

    # --- descriptive artefacts ------------------------------------------
    full = load_dataset(columns=["exit_next_year", "exit_provisional", "vulnerable",
                                 "net_asset_change_3y", "total_revenue",
                                 "is_section_d", "charity_type"])
    summary = outcome_summary(full)
    _write(summary, "outcome_base_rates")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    plots.plot_exit_rates(summary, ax=ax)
    plots.save_figure(fig, "exit_rate_by_year")
    plt.close(fig)
    del full

    tasks: dict[str, TaskResult] = {}

    # --- exit task -------------------------------------------------------
    frame = load_dataset(years=[TRAIN_YEAR, TEST_YEAR, SCORE_YEAR], columns=columns)
    train, test = exit_split(frame)
    tasks["exit"] = run_task("exit", "exit_next_year", train, test)

    # The Tuckman-Chang score's response curve: does one more flag mean more risk?
    tc_model = tasks["exit"].fitted["tuckman_chang_1991"]
    scored = tasks["exit"].test.assign(
        tc_score=tc_model.estimator.score(tasks["exit"].test)
    )
    _write(scored.groupby("tc_score", observed=True)["exit_next_year"]
           .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "exit_rate"}),
           "exit_tuckman_chang_response")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    plots.plot_score_response(scored, "tc_score", "exit_next_year", ax=ax,
                             title="Tuckman-Chang score against realised exit")
    plots.save_figure(fig, "exit_tuckman_chang_response")
    plt.close(fig)

    run_watchlist(tasks["exit"], frame)
    del frame, train, test

    # --- vulnerability task ---------------------------------------------
    if not skip_vulnerability:
        frame = load_dataset(years=[2019, 2020], columns=columns)
        train, test = vulnerability_split(frame)
        tasks["vulnerability"] = run_task("vulnerability", "vulnerable", train, test)

        for name, spec in [("greenlee_trussel_2000", GREENLEE_TRUSSEL_2000),
                           ("trussel_2002", TRUSSEL_2002)]:
            fit = fit_statsmodels_logit(train, "vulnerable", spec["numeric"],
                                        spec["categorical"])
            _write(fit.summary2().tables[1], f"vulnerability_coefficients_{name}")
        del frame, train, test

    log.info("pipeline finished in %.1f s", time.time() - started)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="rebuild the panel and analysis frame from the raw CSVs")
    parser.add_argument("--skip-vulnerability", action="store_true",
                        help="run only the exit task")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    run(refresh=args.refresh, skip_vulnerability=args.skip_vulnerability)


if __name__ == "__main__":
    main()
