"""Inspect the raw RecipeNLG dataset: columns, size, samples, field notes.

Run:  python -m scripts.explore_dataset [--limit 5000]
"""

from __future__ import annotations

import argparse
import textwrap

import pandas as pd

from src.data.loader import load_raw_recipes
from src.utils.text import parse_list_field

RULE = "=" * 72


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_columns(df: pd.DataFrame) -> None:
    section("1. COLUMNS")
    for name in df.columns:
        non_null = df[name].notna().sum()
        example = str(df[name].iloc[0])[:60].replace("\n", " ")
        print(f"  {name:<14} dtype={str(df[name].dtype):<8} "
              f"non-null={non_null:<8} e.g. {example}...")


def show_size(df: pd.DataFrame) -> None:
    section("2. SIZE")
    print(f"  rows      : {len(df):,}")
    print(f"  columns   : {df.shape[1]}")
    print(f"  memory    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("  note      : this is the loaded sample; the full RecipeNLG "
          "dataset has ~2,231,142 rows.")


def show_samples(df: pd.DataFrame, n: int = 5) -> None:
    section(f"3. {n} SAMPLE RECORDS")
    for _, row in df.head(n).iterrows():
        print(f"\n  TITLE       : {row.get('title')}")
        for field in ("ingredients", "NER", "ner", "directions"):
            if field not in row:
                continue
            items = parse_list_field(row[field])
            print(f"  {field.upper():<12}: {len(items)} items")
            for item in items[:3]:
                print(textwrap.fill(str(item), width=68,
                                    initial_indent="      - ",
                                    subsequent_indent="        "))
            if len(items) > 3:
                print(f"      ... (+{len(items) - 3} more)")
        print(f"  SOURCE      : {row.get('source')}")


def show_field_choice(df: pd.DataFrame) -> None:
    section("4. FIELDS TO USE FOR SEARCH")
    verdict = [
        ("NER", "USE (weight 3)",
         "bare ingredient names, already stripped of quantities/units - "
         "matches how users type"),
        ("title", "USE (weight 2)",
         "carries dish intent ('banana bread') that ingredients cannot"),
        ("ingredients", "USE (weight 1)",
         "catches wording NER drops, but noisy with '1 c.', 'chopped'"),
        ("directions", "DISPLAY ONLY",
         "long and procedural; would swamp the ingredient signal"),
        ("link / source", "METADATA",
         "shown for attribution, never vectorized"),
    ]
    for field, decision, why in verdict:
        print(f"  {field:<14} {decision:<16} {why}")

    if "NER" in df.columns or "ner" in df.columns:
        col = "NER" if "NER" in df.columns else "ner"
        counts = df[col].map(lambda v: len(parse_list_field(v)))
        print(f"\n  ingredients per recipe (NER): "
              f"min={counts.min()} median={int(counts.median())} "
              f"max={counts.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5000,
                        help="rows to load (0 = all; default 5000)")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    df = load_raw_recipes(limit=args.limit or None)

    show_columns(df)
    show_size(df)
    show_samples(df, args.samples)
    show_field_choice(df)
    print()


if __name__ == "__main__":
    main()
