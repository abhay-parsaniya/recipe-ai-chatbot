"""Small text helpers shared by loading, preprocessing and (later) search."""

import ast
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^a-z0-9\s]")


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
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = _NON_TEXT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
