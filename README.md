# charity_risk

Predicting organisational failure in the Canadian charity sector from CRA T3010
returns, benchmarked against the classical nonprofit financial-vulnerability
literature — Tuckman and Chang (1991), Greenlee and Trussel (2000), Trussel
(2002) — and against the deep-learning approach of Alam, Gao and Jones (2021).

## What the project finds

Fitted on the 2020 filing cohort, tested strictly out of time on 2021
(84,344 / 84,339 charity-years; 2.0% exit base rate):

| Model | ROC-AUC | Avg. precision | Lift @ top 1% | Recall @ top 5% |
|---|---|---|---|---|
| Model-average ensemble (logit + GBM + net) | 0.844 | 0.195 | 16.3x | 40.8% |
| Gradient boosting (extended) | 0.842 | 0.188 | 16.3x | 40.3% |
| **Deep neural net (extended)** | 0.828 | 0.174 | 14.8x | 38.6% |
| Elastic-net logit (extended) | 0.827 | 0.155 | 12.3x | 38.2% |
| **Trussel (2002)** | 0.807 | 0.111 | 9.2x | 35.4% |
| **Greenlee-Trussel (2000)** | 0.744 | 0.094 | 9.1x | 33.5% |
| Size only (log revenue) | 0.786 | 0.081 | 5.9x | 28.0% |
| **Tuckman-Chang score (0-4)** | 0.587 | 0.032 | 4.2x | 14.9% |

Seven things worth pulling out:

- **The Tuckman-Chang measures still work; the scoring rule does not.** Charities
  flagged on all four measures deregister at 12.1% against 1.7% for those flagged
  on none — the ratios sort risk thirty years on, in a different country. But
  only 430 of 84,000 charities score 4, and 79% of the register falls into two
  buckets with indistinguishable exit rates. Discretising into a 0-4 count throws
  away most of the signal.
- **A single variable nearly matches a published model.** Log revenue alone
  reaches AUC 0.786, against 0.807 for the full Trussel (2002) specification with
  sector fixed effects. Any claim that a ratio set predicts failure needs a size
  baseline next to it.
- **The administrative cost ratio points the wrong way.** Higher administrative
  spending predicts deregistration, not lower. Tuckman and Chang's slack argument
  — lean overhead means nothing left to cut — does not survive this outcome, and
  their scoring rule flags the wrong tail because of it.
- **Failure is organisation-specific.** A hierarchical model puts the between-
  category and between-province standard deviations at 0.28 and 0.22 log-odds,
  against roughly a twenty-fold range across organisational predictors.
- **Deep learning is competitive, not superior.** A three-layer network after
  Alam, Gao and Jones (2021) places third of eight — ahead of every classical
  benchmark, midway between the elastic-net logit and gradient boosting on
  average precision, and indistinguishable from the logit on AUC. Their finding
  that deep learning beats traditional panel models does not reproduce here;
  84,000 annual charity-years is not 641,667 firm-months. Architecture was
  chosen by cross-validation on the *training* year only, which costs about
  0.011 average precision relative to picking on the test year.
- **Combining models is worth the trouble, mainly for calibration.** Averaging
  the elastic-net logit's, gradient boosting's and the neural net's predicted
  probabilities tops the table on both metrics, and is a materially
  better-calibrated score than any of its parts (slope 0.95 against gradient
  boosting's 0.80). The ranking gain over gradient boosting alone is real but
  narrow — average precision +0.007, a bootstrap interval that only just clears
  zero — so a supervisor who cares purely about ordering the register might
  reasonably stick with the single, cheaper model.
- **The detailed return is a substitute for model flexibility, not a complement
  to it.** Restricted to the ~57,000 charity-years filed on Schedule 6, adding
  50 itemised balance-sheet, cost-structure and disbursement-quota features
  lifts an elastic-net logit's average precision by 15% — and moves gradient
  boosting from 0.145 to 0.145. If your model can bend, you do not need the long
  form to triage well.

Full results, figures and caveats are in
[`notebooks/03-model-results.ipynb`](notebooks/03-model-results.ipynb).

## Getting started

```bash
pixi run test
```

```bash
pixi run pipeline-refresh
```

The first builds the panel and analysis frame from `data-raw/t3010` (~10 minutes,
~250 MB of parquet under `data/`), fits every model, and writes tables to
`outputs/tables` and figures to `outputs/figures`. Subsequent runs can use the
cache:

```bash
pixi run pipeline
```

In a notebook or script:

```python
from charity_risk.dataset import load_dataset, exit_split
from charity_risk.evaluate import fit_all, compare_models

frame = load_dataset(years=[2020, 2021])
train, test = exit_split(frame)
compare_models(fit_all(train, "exit_next_year"), test, "exit_next_year")
```

## Notebooks

Read in order.

| | |
|---|---|
| [`00-literature-review`](notebooks/00-literature-review.ipynb) | The five benchmark studies, and what changes when you move to Canadian data |
| [`01-data-and-outcomes`](notebooks/01-data-and-outcomes.ipynb) | The T3010 extract, the charity-year panel, and what "failure" means here |
| [`02-features-and-benchmarks`](notebooks/02-features-and-benchmarks.ipynb) | The Tuckman-Chang measures and both logit replications, re-estimated |
| [`03-model-results`](notebooks/03-model-results.ipynb) | The out-of-time comparison, a deep network, a forecast-average ensemble, a Bayesian alternative, a watchlist, and the limits |
| [`04-schedule-6-deep-dive`](notebooks/04-schedule-6-deep-dive.ipynb) | Section D filings excluded; what the detailed return's 50 extra features actually buy |
| [`05-implementation-and-extensions`](notebooks/05-implementation-and-extensions.ipynb) | Why this is useful, two ways to deploy it, and what better data would buy — no cohort is fitted here |
| [`06-multidimensional-risk-index`](notebooks/06-multidimensional-risk-index.ipynb) | A Tuckman-Chang-style index over twelve ratios in five dimensions, scored against the rule and the logits |

## Package layout

`src/charity_risk/`, each module depending only on those above it:

| Module | Responsibility |
|---|---|
| `config` | Paths, the observation window, the temporal split |
| `fields` | T3010 line codes, and which filing tier is asked for each |
| `ingest` | Raw CSV readers with a parquet cache; absorbs the 2023 schema drift |
| `panel` | The charity-year panel: de-duplication, tier-aware blanks, filing presence |
| `outcomes` | Permanent exit, and the Greenlee-Trussel net-asset decline |
| `features` | Ratios, from the Tuckman-Chang four outwards, in named groups (including the Schedule-6-only group) |
| `dataset` | The modelling frame and the train / test / score / Schedule-6 splits |
| `benchmarks` | The classical models, reproduced, plus statsmodels coefficient tables |
| `risk_index` | A multi-dimensional, Tuckman-Chang-style index over the HS ratios |
| `models` | Preprocessing pipelines and the specification registry |
| `evaluate` | Fitting, metrics, paired bootstrap, permutation importance |
| `hierarchical` | A partially-pooled Bayesian logit (PyMC) |
| `plots` | The figure system |
| `pipeline` | The end-to-end run |

## Two decisions that shape everything downstream

**A blank cell means two different things.** About a third of Canadian
charity-years are filed on the short Section D form, which never asks for cash,
payables, investment income, compensation or fundraising expense. A blank there
is *missing*; the same blank on a Schedule 6 return is *nil*. Zero-filling both
would turn the filing tier into a financial ratio. `fields.is_collected` is the
single place that distinction lives, and it is verified against the raw extract
in `01-data-and-outcomes`.

**Disappearance is a proxy, not a verdict.** A registered charity must file
annually, so a BN that stops appearing has been deregistered — but that bundles
collapse together with solvent wind-ups, amalgamations, and revocations for
failure to file. Every result here predicts *deregistration*. Linking to the
CRA's published revocation reasons would separate those cases and is the single
highest-value extension available.

## A note on the two samples

Results come from two populations, and mixing them up would be easy and wrong.

The headline models run on **every filer**, using only fields both filing tiers
report. `04-schedule-6-deep-dive` runs on **detailed-return filers only** —
about two-thirds of the register, but under half of the exits, because
Schedule 6 is the form you file once you outgrow the small-charity thresholds.
Its exit base rate is 1.2% against 2.0%, so its metrics are not comparable with
the headline table in either direction. `dataset.schedule_6_split` and
`models.SCHEDULE_6_SPECIFICATIONS` are kept separate from the main pipeline for
exactly this reason.

## Data

`data-raw/t3010/{2019..2023}/`, the CRA public T3010 extract, plus the official
data dictionary under `docs/`. Only `Ident.csv` and
`Financial Section D & Schedule 6.csv` are used — they are the two files present
in all five years. Sources for the benchmark literature are in
[`notebooks/refs.bib`](notebooks/refs.bib), with PDFs in `reading/`.
