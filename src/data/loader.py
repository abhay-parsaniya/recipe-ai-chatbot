"""Load the RecipeNLG dataset into a pandas DataFrame.

Two paths are supported, tried in this order:

1. A local CSV (``full_dataset.csv``) under ``dataset_dir``.
2. The Hugging Face hub loader ``mbien/recipe_nlg``.

Why both: ``mbien/recipe_nlg`` is a *manual download* dataset. Its loading
script does not download anything -- RecipeNLG is distributed from
https://recipenlg.cs.put.poznan.pl/ under its own licence, and you must
fetch and unzip it yourself. The HF path therefore only works once the CSV
is already on disk, which is exactly when the faster CSV path works too.
The hub path is kept so the code still matches the stated tech stack and
keeps working if the dataset is ever republished as plain parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Filenames the RecipeNLG archive is commonly extracted to.
CSV_CANDIDATES = ("full_dataset.csv", "RecipeNLG_dataset.csv", "recipe_nlg.csv")

DOWNLOAD_HINT = (
    "RecipeNLG must be downloaded manually from "
    "https://recipenlg.cs.put.poznan.pl/ (accept the licence, unzip), then "
    "place `full_dataset.csv` in `{dir}` or set DATASET_DIR in your .env."
)


def find_local_csv(dataset_dir: Path) -> Path | None:
    """Return the RecipeNLG CSV under ``dataset_dir``, if present."""
    for name in CSV_CANDIDATES:
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate
    # Fall back to any single CSV in the directory.
    csvs = sorted(dataset_dir.glob("*.csv"))
    return csvs[0] if len(csvs) == 1 else None


def load_from_csv(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Read the CSV directly.

    ``nrows`` is used instead of reading everything and slicing, so a
    50k-row sample costs a couple of seconds rather than minutes.

    ``index_col=0``: RecipeNLG ships with an unnamed row-number column.
    Left alone pandas calls it "Unnamed: 0" and it travels through the
    whole pipeline as a junk feature.
    """
    logger.info("Reading %s (limit=%s)", path, limit or "none")
    return pd.read_csv(path, nrows=limit, index_col=0)


def load_from_hub(settings: Settings, limit: int | None = None) -> pd.DataFrame:
    """Load through `datasets`, streaming when a limit is set.

    Streaming avoids materialising 2.2M rows just to keep the first 50k.
    """
    from datasets import load_dataset  # imported lazily: heavy import

    logger.info("Loading %s from the Hugging Face hub", settings.dataset_name)
    kwargs = {
        "path": settings.dataset_name,
        "split": settings.dataset_split,
        "trust_remote_code": True,
    }
    if settings.dataset_dir.exists():
        kwargs["data_dir"] = str(settings.dataset_dir)

    if limit:
        stream = load_dataset(**kwargs, streaming=True)
        rows = list(stream.take(limit))
        return pd.DataFrame(rows)

    return load_dataset(**kwargs).to_pandas()


def load_raw_recipes(
    limit: int | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Entry point used by scripts and the preprocessing module.

    Three distinct meanings, because two were not enough:

    * ``limit=None`` (the default) -- use the MAX_RECIPES setting.
    * ``limit=0``    -- no cap, load everything.
    * ``limit=N``    -- load N rows.

    An earlier version used ``None`` for "no cap" as well as "use the
    setting", so ``--limit 0`` on the command line collapsed to ``None``
    and silently re-applied the 50,000 cap. There was no way to ask for
    the full corpus at all.
    """
    settings = settings or get_settings()
    if limit is None:
        limit = settings.max_recipes or None
    elif limit == 0:
        limit = None

    csv_path = find_local_csv(settings.dataset_dir)
    if csv_path is not None:
        return load_from_csv(csv_path, limit)

    logger.warning("No local CSV in %s; trying the hub", settings.dataset_dir)
    try:
        return load_from_hub(settings, limit)
    except Exception as exc:  # noqa: BLE001 - surface a usable message
        raise FileNotFoundError(
            f"Could not load {settings.dataset_name}: {exc}\n"
            + DOWNLOAD_HINT.format(dir=settings.dataset_dir)
        ) from exc
