"""Measure retrieval quality against evals/search_cases.json.

Why mechanical relevance instead of human labels: the corpus has 390k
recipes and no relevance judgements, and hand-labelling even 40 queries
is a day's work that goes stale the moment the index changes. Each case
instead declares what a relevant result *looks like* -- "contains all of
chicken, tomato, onion", "title mentions cacciatore" -- and the runner
checks the returned recipes against their own fields.

That is weaker than human judgement: it cannot tell a good chicken
recipe from a dull one. It is strong enough for the thing an eval is
actually for here -- telling whether a change to ranking, corpus size or
scoring made retrieval better or worse, rather than trusting a hunch
about one query.

Run:  python -m scripts.eval_search [--top-k 5] [--save baseline.json]
      python -m scripts.eval_search --compare baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.config.settings import get_settings
from src.search import RecipeSearch
from src.search.pantry import PantryMatcher
from src.utils.text import normalize

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "search_cases.json"


def haystack(result) -> str:
    """Everything a relevance predicate may look at, normalized once."""
    return normalize(" ".join(
        [result.title] + list(result.ner) + list(result.ingredients)
    ))


def is_relevant(case: dict, result) -> bool:
    if "require_all" in case:
        text = haystack(result)
        return all(normalize(term) in text for term in case["require_all"])
    if "title_any" in case:
        title = normalize(result.title)
        return any(normalize(term) in title for term in case["title_any"])
    raise ValueError(f"case {case['id']} declares no relevance predicate")


def run_search_cases(engine: RecipeSearch, cases: list[dict],
                     top_k: int) -> dict:
    rows, latencies = [], []

    for case in cases:
        started = time.perf_counter()
        results = engine.search(case["query"], top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        flags = [is_relevant(case, r) for r in results]
        hits = sum(flags)
        first = next((i + 1 for i, ok in enumerate(flags) if ok), 0)
        titles = [r.title for r in results]

        rows.append({
            "id": case["id"],
            "kind": case["kind"],
            "query": case["query"],
            "returned": len(results),
            "precision_at_k": hits / top_k,
            "recall_any": 1.0 if hits else 0.0,
            "reciprocal_rank": 1 / first if first else 0.0,
            "distinct_titles": len(set(titles)) / len(titles) if titles else 0.0,
            "top_title": titles[0] if titles else "",
        })

    def mean(field):
        return statistics.mean(r[field] for r in rows) if rows else 0.0

    return {
        "cases": rows,
        "summary": {
            "n": len(rows),
            "precision_at_k": mean("precision_at_k"),
            "answered": mean("recall_any"),
            "mrr": mean("reciprocal_rank"),
            "distinct_titles": mean("distinct_titles"),
            "empty_results": sum(1 for r in rows if r["returned"] == 0),
            "latency_ms_median": statistics.median(latencies) if latencies else 0,
            "latency_ms_p95": (sorted(latencies)[int(len(latencies) * 0.95)]
                               if latencies else 0),
        },
    }


def run_pantry_cases(matcher: PantryMatcher, cases: list[dict],
                     top_k: int) -> dict:
    rows = []
    for case in cases:
        started = time.perf_counter()
        matches = matcher.match(have=case["have"], exclude=case["exclude"],
                                top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000

        # An exclusion that leaks is a correctness bug, not a quality
        # one, so it is counted separately and must always be zero.
        leaks = sum(
            1 for m in matches
            for banned in case["exclude"]
            if normalize(banned) in normalize(" ".join(m.ingredients))
        )
        rows.append({
            "id": case["id"],
            "have": case["have"],
            "returned": len(matches),
            "complete": sum(1 for m in matches if m.is_complete),
            "best_match": max((m.match for m in matches), default=0.0),
            "mean_missing": (statistics.mean(len(m.missing) for m in matches)
                             if matches else 0.0),
            "exclusion_leaks": leaks,
            "latency_ms": elapsed,
        })

    def mean(field):
        return statistics.mean(r[field] for r in rows) if rows else 0.0

    return {
        "cases": rows,
        "summary": {
            "n": len(rows),
            "best_match": mean("best_match"),
            "complete_per_case": mean("complete"),
            "mean_missing": mean("mean_missing"),
            "exclusion_leaks": sum(r["exclusion_leaks"] for r in rows),
            "empty_results": sum(1 for r in rows if r["returned"] == 0),
            "latency_ms_median": statistics.median(
                [r["latency_ms"] for r in rows]) if rows else 0,
        },
    }


def print_report(report: dict) -> None:
    search, pantry = report["search"]["summary"], report["pantry"]["summary"]
    print(f"\ncorpus: {report['corpus']:,} recipes   top_k={report['top_k']}")
    print("=" * 62)
    print("SEARCH")
    print(f"  precision@k      {search['precision_at_k']:.3f}"
          "   (relevant results per page)")
    print(f"  answered         {search['answered']:.3f}"
          "   (cases with >=1 relevant hit)")
    print(f"  MRR              {search['mrr']:.3f}"
          "   (how high the first hit lands)")
    print(f"  distinct titles  {search['distinct_titles']:.3f}")
    print(f"  empty results    {search['empty_results']}")
    print(f"  latency          {search['latency_ms_median']:.0f} ms median, "
          f"{search['latency_ms_p95']:.0f} ms p95")

    print("\nPANTRY")
    print(f"  best match       {pantry['best_match']:.3f}")
    print(f"  complete/case    {pantry['complete_per_case']:.2f}")
    print(f"  mean missing     {pantry['mean_missing']:.2f} items")
    print(f"  exclusion leaks  {pantry['exclusion_leaks']}  (must be 0)")
    print(f"  latency          {pantry['latency_ms_median']:.0f} ms median")

    worst = sorted(report["search"]["cases"], key=lambda r: r["precision_at_k"])
    print("\nweakest cases:")
    for row in worst[:6]:
        print(f"  {row['precision_at_k']:.2f}  {row['id']:<8} "
              f"{row['query'][:38]:<38} -> {row['top_title'][:28]}")


def compare(current: dict, baseline: dict) -> None:
    print("\nCOMPARED TO BASELINE")
    print("=" * 62)
    print(f"  corpus  {baseline['corpus']:,} -> {current['corpus']:,}")
    for section in ("search", "pantry"):
        for metric, value in current[section]["summary"].items():
            before = baseline[section]["summary"].get(metric)
            if not isinstance(value, (int, float)) or before is None:
                continue
            delta = value - before
            arrow = "=" if abs(delta) < 1e-9 else ("+" if delta > 0 else "")
            print(f"  {section:<7} {metric:<20} {before:>9.3f} -> "
                  f"{value:>9.3f}  ({arrow}{delta:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    spec = json.loads(CASES_PATH.read_text())
    settings = get_settings()
    engine = RecipeSearch.load(settings=settings)

    report = {
        "corpus": len(engine.recipes),
        "top_k": args.top_k,
        "search": run_search_cases(engine, spec["cases"], args.top_k),
        "pantry": run_pantry_cases(PantryMatcher(engine, settings),
                                   spec["pantry_cases"], args.top_k),
    }
    print_report(report)

    if args.compare and args.compare.exists():
        compare(report, json.loads(args.compare.read_text()))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2))
        print(f"\nsaved to {args.save}")


if __name__ == "__main__":
    main()
