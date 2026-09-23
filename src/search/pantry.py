"""Ingredient-based recommendation: "what can I actually cook tonight?"

This asks a different question from `/search`, and the difference is the
whole point of the feature.

* Search coverage = *how much of your query* the recipe contains. Good
  for "find me something with chicken and tomato".
* Pantry coverage = *how much of the recipe* you already have. That is
  the number that decides whether you can cook it without shopping.

A recipe matching all three words you typed is useless if it needs nine
other things you do not own. So this module scores on recipe
completeness, reports the missing items by name, and lets you rule
ingredients out entirely.

Matching is approximate, and honestly so. "chicken" matches "chicken
breasts" because the user means they have chicken. The same rule means
"tomato" matches "tomato sauce", which is sometimes wrong -- having a
tomato is not having a jar of sauce. The missing list is always shown so
the user can see what the match actually assumed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.config.settings import Settings, get_settings
from src.search.engine import RecipeSearch
from src.utils.logging import get_logger
from src.utils.text import normalize, prettify_title

logger = get_logger(__name__)

# Things almost every kitchen has. Counting them as "missing" would rank
# a two-ingredient recipe below a ten-ingredient one purely because the
# short one lists salt. Assumed by default; switch off per request.
STAPLES = {
    "salt", "water", "pepper", "black pepper", "sugar", "flour", "oil",
    "olive oil", "vegetable oil", "cooking oil", "butter", "margarine",
    "ice", "ice water", "hot water", "cold water", "salt and pepper",
}

_PLURAL = re.compile(r"(ies|es|s)$")


def singularize(word: str) -> str:
    """Crude but predictable: "tomatoes" -> "tomato", "berries" -> "berri".

    Good enough for matching because both sides go through it, so the
    two spellings meet in the middle even when the middle is not a real
    word.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("ses", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize_ingredient(text: str) -> set[str]:
    return {singularize(token) for token in normalize(text).split() if token}


@dataclass
class PantryMatch:
    """One recipe, scored by how much of it the user already has."""

    recipe_id: int
    title: str
    match: float                  # have / needed, 0-1
    have: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    assumed_staples: list[str] = field(default_factory=list)
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    link: str = ""

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        data = asdict(self)
        data["is_complete"] = self.is_complete
        return data


class PantryMatcher:
    """Rank recipes by how cookable they are from a given ingredient list.

    Wraps a fitted RecipeSearch rather than replacing it: TF-IDF picks
    the candidate pool (fast, over 46k rows), then every candidate is
    scored exactly in Python. Scoring all 46k rows exactly would mean a
    set intersection per recipe per request; scoring a few hundred is
    instant and the recipes TF-IDF misses are ones sharing none of the
    user's ingredients, which could not score well anyway.
    """

    def __init__(self, engine: RecipeSearch, settings: Settings | None = None):
        self.engine = engine
        self.settings = settings or get_settings()

    def match(
        self,
        have: list[str],
        exclude: list[str] | None = None,
        top_k: int | None = None,
        min_match: float = 0.0,
        assume_staples: bool = True,
        candidate_pool: int | None = None,
    ) -> list[PantryMatch]:
        """Recipes you can mostly make, best first."""
        have = [item for item in (have or []) if item.strip()]
        if not have:
            return []

        exclude = [item for item in (exclude or []) if item.strip()]
        top_k = top_k or self.settings.default_top_k
        pool = candidate_pool or self.settings.pantry_candidate_pool

        have_tokens = [tokenize_ingredient(item) for item in have]
        exclude_tokens = [tokenize_ingredient(item) for item in exclude]

        # max_per_title=0: the pantry needs every candidate it can get,
        # and it dedups nothing -- two different "Banana Bread" recipes
        # need different things from your cupboard.
        candidates = self.engine.search(" ".join(have), top_k=pool,
                                        max_per_title=0)
        logger.info("Pantry: %d candidates for %d ingredients",
                    len(candidates), len(have))

        matches: list[PantryMatch] = []
        for result in candidates:
            needed = result.ner or result.ingredients
            if not needed:
                continue
            scored = self._score(result, needed, have, have_tokens,
                                 exclude_tokens, assume_staples)
            if scored is not None and scored.match > min_match:
                matches.append(scored)

        # Ties on match fraction are broken by recipe size: with two
        # recipes you can fully make, the one needing more of your
        # ingredients is the more interesting suggestion.
        matches.sort(key=lambda m: (m.match, len(m.have)), reverse=True)
        return matches[:top_k]

    def _score(self, result, needed: list[str], have: list[str],
               have_tokens: list[set[str]], exclude_tokens: list[set[str]],
               assume_staples: bool) -> PantryMatch | None:
        matched_have: list[str] = []
        missing: list[str] = []
        assumed: list[str] = []

        for item in needed:
            item_tokens = tokenize_ingredient(item)
            if not item_tokens:
                continue

            if any(self._overlaps(item_tokens, tokens)
                   for tokens in exclude_tokens):
                return None          # the user ruled this recipe out

            owner = self._owner(item_tokens, have_tokens, have)
            if owner is not None:
                if owner not in matched_have:
                    matched_have.append(owner)
                continue

            if assume_staples and normalize(item) in STAPLES:
                assumed.append(item)
                continue

            missing.append(item)

        countable = len(matched_have) + len(missing)
        if countable == 0:
            # Everything was a staple. Cookable, but not evidence that
            # the user's ingredients are involved -- skip it.
            return None

        return PantryMatch(
            recipe_id=result.recipe_id,
            title=prettify_title(result.title),
            match=round(len(matched_have) / countable, 4),
            have=matched_have,
            missing=missing,
            assumed_staples=assumed,
            ingredients=result.ingredients,
            directions=result.directions,
            link=result.link,
        )

    @staticmethod
    def _overlaps(item_tokens: set[str], user_tokens: set[str]) -> bool:
        """True if a user ingredient names this recipe ingredient.

        Requires every word of the user's term to appear in the recipe's
        term -- "olive oil" matches "extra virgin olive oil" but plain
        "oil" does not match "oil of oregano" the other way round, and
        "cream" does not silently satisfy "cream cheese"... except that
        it does, since "cream" is a subset. That over-match is the
        documented tradeoff; the missing list keeps it visible.
        """
        return bool(user_tokens) and user_tokens <= item_tokens

    @classmethod
    def _owner(cls, item_tokens: set[str], have_tokens: list[set[str]],
               have: list[str]) -> str | None:
        for original, tokens in zip(have, have_tokens):
            if cls._overlaps(item_tokens, tokens):
                return original
        return None
