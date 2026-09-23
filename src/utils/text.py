"""Small text helpers shared by loading, preprocessing and (later) search."""

import ast
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^a-z0-9\s]")
# Apostrophes are deleted, not replaced with a space, so "Shepherd's"
# becomes "shepherds" and matches a user typing "shepherds pie".
# Replacing it with a space produced the token pair "shepherd" + "s",
# which no apostrophe-free query could ever hit -- 226 Shepherd's Pie
# recipes were unreachable unless you typed the apostrophe yourself.
_APOSTROPHE = re.compile(r"[\u2019'`]")


def parse_list_field(value) -> list[str]:
    """RecipeNLG stores lists as stringified JSON, e.g. '["1 cup sugar"]'.

    pandas reads those as plain strings, so every consumer would otherwise
    re-invent this parsing. Returns [] rather than raising on bad rows --
    a handful of malformed records should not kill a 2M-row load.
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(v).strip() for v in parsed if str(v).strip()]
    return [str(parsed).strip()]


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Used to build the field we vectorize. Keeping it dumb and
    deterministic means the same function can normalize a user query at
    query time -- query and index must go through identical treatment or
    similarity scores are meaningless.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = _APOSTROPHE.sub("", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = _NON_TEXT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


# RecipeNLG title-cased every title mechanically, which mangles
# possessives ("Ball'S") and glues brackets to the preceding word
# ("Cups(Candy)"). These are display-only repairs -- matching, dedup and
# the title bonus all run on normalize(), which strips punctuation
# anyway, so prettifying cannot change any ranking.
_POSSESSIVE = re.compile(r"(\w)'S\b")
_TIGHT_BRACKET = re.compile(r"(\w)\(")
_ACRONYMS = {"Bbq": "BBQ", "Tv": "TV", "Blt": "BLT", "Pb": "PB"}


def prettify_title(title: str) -> str:
    """Make a RecipeNLG title fit to show a human.

    "Jewell Ball'S Chicken"        -> "Jewell Ball's Chicken"
    "Quicky Chicken(Serves 4)  "   -> "Quicky Chicken (Serves 4)"
    """
    if not title:
        return ""
    text = _POSSESSIVE.sub(r"\1's", str(title))
    text = _TIGHT_BRACKET.sub(r"\1 (", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return " ".join(_ACRONYMS.get(word, word) for word in text.split())
