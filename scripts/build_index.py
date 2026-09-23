"""Fit the TF-IDF index over processed recipes and save it to disk.

Run:  python -m scripts.build_index [--query "chicken rice"]
"""

from __future__ import annotations

import argparse

from src.search import RecipeSearch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="chicken rice garlic onion",
                        help="smoke-test query run against the new index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    engine = RecipeSearch.from_processed()
    path = engine.save()
    print(f"Saved index to {path}")

    print(f"\nSmoke test: {args.query!r}")
    for rank, result in enumerate(engine.search(args.query, args.top_k), 1):
        print(f"  {rank}. [{result.score:.4f}] {result.title}")
        print(f"       matched: {', '.join(result.matched_terms) or '-'}")


if __name__ == "__main__":
    main()
