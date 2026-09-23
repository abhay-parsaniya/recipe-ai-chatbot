"""Turn the raw RecipeNLG table into a clean, search-ready DataFrame.

Output schema (one row per recipe):

    recipe_id     int    stable row id
    title         str    recipe name
    ingredients   list   full ingredient lines, with quantities
    directions    list   ordered cooking steps
    ner           list   bare ingredient names ("sugar", "chicken")
    link          str    source URL
    source        str    provenance tag from the dataset
    search_text   str    normalized text the vectorizer will consume

Which fields feed search, and why:

  * ``NER`` is the backbone. It is RecipeNLG's own extraction of the bare
    food entities, with quantities, units and prep words already stripped.
    Users type "chicken, rice, garlic" -- that is exactly the NER
    vocabulary, so matching against it is nearly noise-free.
  * ``title`` is added because intent is often a dish, not an ingredient
    ("banana bread"), and the title is the only field carrying that.
  * ``ingredients`` is added at lower weight: it catches phrasing the NER
    list drops ("buttermilk", "self-rising flour").
  * ``directions`` is deliberately excluded. It is long, procedural and
    dominated by words like "bake", "stir", "minutes"; including it would
    swamp the ingredient signal and slow vectorization for no gain. It is
    still kept in the output because the chatbot must display it.

Weighting is done by repeating a field in ``search_text`` rather than by
post-hoc score maths -- with TF-IDF, repeating a field is a weight, and it
keeps the whole scheme visible in one string.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger
from src.utils.text import normalize, parse_list_field

logger = get_logger(__name__)

# Raw column -> our column. RecipeNLG ships `NER`; guard for `ner` too.
COLUMN_ALIASES = {
    "title": "title",
    "ingredients": "ingredients",
    "directions": "directions",
    "NER": "ner",
    "ner": "ner",
    "link": "link",
    "source": "source",
}

LIST_COLUMNS = ("ingredients", "directions", "ner")

# How many times each field is repeated inside search_text.
FIELD_WEIGHTS = {"ner": 3, "title": 2, "ingredients": 1}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    return df.rename(columns=mapping)


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in LIST_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(parse_list_field)
    return df


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be recommended or displayed.

    A recipe with no title, no ingredients or no steps is dead weight in
    the index: it can never be a useful answer, but it still costs a row
    in the similarity matrix.
    """
    before = len(df)
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]
    df = df[df["ingredients"].map(len) > 0]
    df = df[df["directions"].map(len) > 0]
    logger.info("Dropped %d unusable rows (%d remain)", before - len(df), len(df))
    return df


def _ingredient_fingerprint(items) -> str:
    """Order-independent identity for an ingredient list.

    Sorted, normalized and joined, so "1 c. Sugar" / "1 cup sugar" and
    two recipes listing the same things in a different order collapse to
    the same key. Quantities are kept -- two cakes differing only in how
    much sugar they use are genuinely different recipes.
    """
    return "|".join(sorted(normalize(item) for item in items))


def drop_duplicate_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """RecipeNLG scrapes many sites, so the same recipe recurs often.

    Two passes, because one key cannot catch both shapes of duplicate:

    1. **Same title, same ingredient set.** The literal reprint. Keyed
       on the whole ingredient list rather than its first item -- the
       old key let "Banana Bread" through six times, because scrapes of
       the same recipe often disagree about which ingredient is listed
       first.
    2. **Same title, same NER set.** Catches reprints whose ingredient
       *lines* were reworded ("2 c. flour" vs "2 cups flour") but whose
       extracted ingredients are identical.

    Titles alone are deliberately not a key: "Chicken Casserole" is a
    name thousands of genuinely different recipes share.
    """
    before = len(df)

    full_key = df["title"].map(normalize) + "||" + \
        df["ingredients"].map(_ingredient_fingerprint)
    df = df[~full_key.duplicated()]

    if "ner" in df.columns:
        ner_key = df["title"].map(normalize) + "||" + \
            df["ner"].map(_ingredient_fingerprint)
        df = df[~ner_key.duplicated()]

    logger.info("Dropped %d duplicate recipes (%d remain)",
                before - len(df), len(df))
    return df


# Kept as an alias: the old name appears in notebooks and scripts.
drop_duplicate_titles = drop_duplicate_recipes


def build_search_text(row: pd.Series) -> str:
    parts: list[str] = []
    for field, weight in FIELD_WEIGHTS.items():
        value = row.get(field)
        if value is None:
            continue
        text = " ".join(value) if isinstance(value, list) else str(value)
        parts.extend([text] * weight)
    return normalize(" ".join(parts))


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame -> clean DataFrame. Pure: no I/O, easy to test."""
    df = rename_columns(df.copy())

    missing = {"title", "ingredients", "directions"} - set(df.columns)
    if missing:
        raise KeyError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = parse_list_columns(df)
    df = drop_unusable(df)
    df = drop_duplicate_recipes(df)

    # A few thousand recipes have an empty NER list (the extractor found
    # nothing). They are kept -- title and ingredients still describe
    # them -- but they will rank lower, which is the right outcome.
    df["search_text"] = df.apply(build_search_text, axis=1)
    # An empty search_text would match nothing and pollute the matrix.
    df = df[df["search_text"].str.strip() != ""]

    df = df.reset_index(drop=True)
    df.insert(0, "recipe_id", df.index)

    keep = [
        "recipe_id",
        "title",
        "ingredients",
        "directions",
        "ner",
        "link",
        "source",
        "search_text",
    ]
    return df[[c for c in keep if c in df.columns]]


def save_processed(df: pd.DataFrame, settings: Settings | None = None):
    """Write parquet: it preserves the list columns, CSV would flatten them."""
    settings = settings or get_settings()
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_recipes_path
    df.to_parquet(path, index=False)
    logger.info("Wrote %d recipes to %s", len(df), path)
    return path


def load_processed(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    path = settings.processed_recipes_path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m scripts.build_dataset"
        )
    return pd.read_parquet(path)
