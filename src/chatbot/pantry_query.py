"""Pull a have-list and a don't-have-list out of a natural message.

Recognises the shapes people actually type:

    I have chicken, onion, tomato and garlic but no cheese
    have: chicken, onion / don't have: cheese
    what can I make with only eggs, flour and milk

The split matters: "no cheese" must become an exclusion, not another
ingredient. Feeding "cheese" into the have-list would recommend exactly
the recipes the user is trying to avoid.
"""

from __future__ import annotations

import re

from src.chatbot.query import STOPWORDS
from src.utils.text import normalize

# Everything after one of these belongs to the exclusion list.
EXCLUSION_SPLIT = re.compile(
    r"\b(?:but\s+)?(?:no|not|without|don'?t\s+have|do\s+not\s+have|"
    r"dont\s+have|haven'?t\s+got|missing|out\s+of|except|excluding|"
    r"no\s+more)\b",
    re.IGNORECASE,
)

# Phrases that introduce the have-list; stripped before splitting.
HAVE_PREFIXES = re.compile(
    r"^\s*(?:i\s+)?(?:have|have\s+got|'?ve\s+got|got|own|there'?s|"
    r"pantry|fridge|kitchen)\s*:?\s*|"
    r"^\s*(?:what\s+can\s+i\s+(?:make|cook)\s+with\s+(?:only\s+)?)|"
    r"^\s*(?:using\s+only\s+|only\s+|with\s+only\s+)",
    re.IGNORECASE,
)

# Phrases where the user is inventorying a kitchen rather than
# describing a dish. Note what is NOT here: "what can I make with ..."
# is the commonest way to phrase an ordinary search, so treating it as
# an inventory hijacked normal queries.
INVENTORY_MARKERS = re.compile(
    r"\bi\s+have\b|\bi'?ve\s+got\b|\bi\s+got\b|"
    r"\bin\s+my\s+(?:fridge|pantry|kitchen|cupboard)\b|"
    r"\b(?:fridge|pantry|kitchen)\s*:|"
    r"\ball\s+i\s+have\b",
    re.IGNORECASE,
)

ONLY_MARKER = re.compile(r"\bonly\b", re.IGNORECASE)

SEPARATORS = re.compile(r",|;|\band\b|\bplus\b|\n|/|\+")

# Measurement words. People type their pantry the way they read a
# recipe -- "1 cup flour" -- and "cup flour" matches nothing. Only
# measures and sizes are stripped; preparation words like "ground" stay,
# because "ground beef" is a different ingredient from "beef".
UNITS = {
    "c", "cup", "cups", "tbsp", "tbs", "tablespoon", "tablespoons",
    "tsp", "teaspoon", "teaspoons", "lb", "lbs", "pound", "pounds",
    "oz", "ounce", "ounces", "g", "gram", "grams", "kg", "ml", "l",
    "litre", "litres", "liter", "liters", "pint", "pints", "quart",
    "quarts", "gallon", "can", "cans", "jar", "jars", "pkg", "package",
    "packages", "packet", "bag", "bags", "box", "boxes", "bottle",
    "carton", "clove", "cloves", "slice", "slices", "stick", "sticks",
    "pinch", "dash", "bunch", "head", "piece", "pieces",
    "large", "small", "medium", "big", "half", "whole",
}

# Fractions and mixed numbers: "1/2", "1 1/2".
NUMERIC = re.compile(r"^\d+([./]\d+)?$")


def split_items(text: str) -> list[str]:
    """Split a list fragment into ingredient names, order preserved."""
    items: list[str] = []
    for chunk in SEPARATORS.split(text or ""):
        cleaned = normalize(chunk)
        words = [w for w in cleaned.split()
                 if w not in STOPWORDS and w not in UNITS
                 and not NUMERIC.match(w)]
        name = " ".join(words).strip()
        if name and name not in items:
            items.append(name)
    return items


def looks_like_pantry_request(message: str) -> bool:
    """Decide between "find me a dish" and "what can I cook tonight".

    Three ways to qualify, each requiring real substance so that
    ordinary searches are never hijacked:

    * an exclusion plus at least two ingredients -- "chicken, potatoes,
      out of onions". One ingredient is not an inventory: "chicken
      recipes with no dairy" is a search with a caveat, and treating
      "dairy" as an excludable ingredient would silently fail anyway,
      since no recipe lists "dairy" by name.
    * "only", which always means a closed list.
    * an explicit inventory phrase ("I have", "in my fridge") with three
      or more items. "I have chicken and rice" is how people phrase a
      search, so two items is not enough.

    Anything else stays a search.
    """
    if not message:
        return False

    have, exclude = parse_pantry(message)
    if not have:
        return False
    if exclude and len(have) >= 2:
        return True
    if ONLY_MARKER.search(message) and len(have) >= 2:
        return True
    return bool(INVENTORY_MARKERS.search(message)) and len(have) >= 3


def parse_pantry(message: str) -> tuple[list[str], list[str]]:
    """Return (have, exclude)."""
    if not message:
        return [], []

    parts = EXCLUSION_SPLIT.split(message, maxsplit=1)
    have_text = parts[0]
    exclude_text = parts[1] if len(parts) > 1 else ""

    have_text = HAVE_PREFIXES.sub("", have_text.strip())
    have = split_items(have_text)
    exclude = split_items(exclude_text)

    # A word cannot be both owned and banned; the exclusion wins,
    # because it is the more specific statement.
    have = [item for item in have if item not in exclude]
    return have, exclude
