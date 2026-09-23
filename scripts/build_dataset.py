"""Load raw RecipeNLG, preprocess it, write data/processed/recipes.parquet.

Run:  python -m scripts.build_dataset [--limit 50000]
"""

from __future__ import annotations

import argparse

from src.config.settings import get_settings
from src.data.loader import load_raw_recipes
from src.data.preprocess import preprocess, save_processed


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=settings.max_recipes,
                        help="rows to load; 0 loads the entire file")
    args = parser.parse_args()

    # Passed through as-is: 0 means "everything", and must not be
    # collapsed to None, which means "use MAX_RECIPES".
    raw = load_raw_recipes(limit=args.limit, settings=settings)
    print(f"Loaded {len(raw):,} raw rows")

    clean = preprocess(raw)
    print(f"Kept  {len(clean):,} clean recipes")

    path = save_processed(clean, settings)
    print(f"Saved {path}")
    print("\nPreview:")
    print(clean[["recipe_id", "title"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
