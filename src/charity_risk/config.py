"""Project paths and global constants.

Every path is derived from :data:`PROJECT_ROOT`, which is located by walking up
from this file until a directory containing ``pyproject.toml`` is found.  This
keeps the package importable from a notebook, a script or an editable install
without any environment variables.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "RAW_DIR",
    "DOCS_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "OUTPUT_DIR",
    "FIGURE_DIR",
    "TABLE_DIR",
    "YEARS",
    "PANEL_START",
    "PANEL_END",
    "TRAIN_YEAR",
    "TEST_YEAR",
    "SCORE_YEAR",
    "MIN_LOOKAHEAD_YEARS",
    "RANDOM_STATE",
    "ensure_dirs",
]


def _find_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fall back to three levels up (src/charity_risk/config.py -> project root).
    return start.parents[2]


PROJECT_ROOT: Path = _find_root(Path(__file__).resolve())

RAW_DIR: Path = PROJECT_ROOT / "data-raw" / "t3010"
DOCS_DIR: Path = RAW_DIR / "docs"
INTERIM_DIR: Path = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
TABLE_DIR: Path = OUTPUT_DIR / "tables"

#: Fiscal-period-end years available in ``data-raw/t3010``.
YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
PANEL_START: int = YEARS[0]
PANEL_END: int = YEARS[-1]

#: A permanent-exit label is only trusted when at least this many later years
#: are observed.  Gap-then-return patterns exist in the panel (see
#: ``notebooks/01-data-and-outcomes``), so a single lookahead year is too noisy
#: to treat as failure.
MIN_LOOKAHEAD_YEARS: int = 2

#: Temporal design.  Features for year *t* use levels at *t* and growth from
#: *t-1*, so the first modellable year is 2020.  ``TEST_YEAR`` is strictly
#: out-of-time.  ``SCORE_YEAR`` has features but no trustworthy label; it is
#: scored to produce a forward-looking watchlist.
TRAIN_YEAR: int = 2020
TEST_YEAR: int = 2021
SCORE_YEAR: int = 2022

RANDOM_STATE: int = 20240607


def ensure_dirs() -> None:
    """Create the derived-data and output directories if they do not exist."""
    for path in (INTERIM_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
