"""Print exactly what would be sent to the LLM for a query. Sends nothing.

Run:  python -m scripts.show_prompt "chicken tomato onion"
"""

from __future__ import annotations

import sys

from src.config.settings import get_settings
from src.llm.prompts import SYSTEM_PROMPT, build_user_message
from src.search import RecipeSearch


def main() -> None:
    query = " ".join(sys.argv[1:]) or "chicken tomato onion"
    settings = get_settings()
    engine = RecipeSearch.load(settings=settings)
    results = engine.search(query, top_k=settings.default_top_k)

    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(SYSTEM_PROMPT)
    print()
    print("=" * 70)
    print("USER MESSAGE (retrieved context + question)")
    print("=" * 70)
    print(build_user_message(query, results))


if __name__ == "__main__":
    main()
