"""Turn structured results into text.

Kept apart from the bot's logic so the wording can change without
touching behaviour, and so tests can assert on decisions rather than on
prose. When an LLM is added it replaces the functions in this file.
"""

from __future__ import annotations

from src.search import SearchResult
from src.utils.text import normalize

GREETING = (
    "Hi! Tell me what ingredients you have and I'll find recipes.\n"
    "For example: \"what can I make with chicken, tomato and onion?\""
)

HELP = (
    "Here's what I can do:\n"
    "  - Find recipes: \"I have chicken, rice and garlic\"\n"
    "  - Pick one:     \"show me the second one\" or just \"2\"\n"
    "  - Then ask:     \"what ingredients do I need?\" / \"how do I make it?\"\n"
    "  - See others:   \"more options\"\n"
    "  - Begin again:  \"start over\""
)

GOODBYE = "Happy cooking!"

NO_RESULTS = (
    "I couldn't find anything matching that. Try naming a few plain "
    "ingredients, like \"chicken, tomato, onion\"."
)

NOTHING_SELECTED = (
    "Pick a recipe first -- say \"2\" or \"show me the first one\"."
)

NO_SEARCH_YET = "Tell me some ingredients first and I'll find recipes."

UNKNOWN = (
    "I'm not sure what you mean. I only know recipes -- "
    "name some ingredients, or say \"help\"."
)

# The canonical refusal. One sentence stating what this assistant is,
# one stating what it can do -- no apology, no lecture, and crucially no
# attempt at the question.
OUT_OF_DOMAIN = (
    "I'm a recipe assistant. I can help you find recipes, ingredients, "
    "cooking instructions, and meal ideas."
)

# Appended per category so the refusal does not read as a wall the user
# has to guess their way around.
OUT_OF_DOMAIN_NUDGES = {
    "politics": "Ask me what to cook instead — I'm better at dinner than at government.",
    "weather": "I can't see outside, but I can suggest something warming.",
    "general_knowledge": "Ask me about an ingredient or a dish and I'll do better.",
    "coding": "Recipes are the only instructions I follow.",
    "medical": "I can't give health advice. For diet or health questions, "
               "ask a professional — I can still find you recipes.",
    "entertainment": "I only know recipes. Want one?",
    "personal": "I'm a search tool over a recipe database, nothing more.",
    "other": "Tell me some ingredients and I'll find recipes.",
}


def format_out_of_domain(category: str = "other") -> str:
    nudge = OUT_OF_DOMAIN_NUDGES.get(category)
    return f"{OUT_OF_DOMAIN}\n\n{nudge}" if nudge else OUT_OF_DOMAIN


NEGATION_CAVEAT = (
    "  (Heads up: I match on words, so I can't reliably exclude things "
    "yet -- please check the ingredients.)"
)


def join_naturally(items: list[str], conjunction: str = "and") -> str:
    """"a, b and c" -- how a person writes a list, not "a, b, c"."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def extra_ingredients(result: SearchResult, query: str,
                      limit: int = 4) -> list[str]:
    """What this recipe needs *beyond* what the user asked about.

    The old listing repeated the user's own words back at them -- ask
    for chicken and tomato and the preview line said "chicken, tomato".
    What actually helps someone choose is the opposite: what else they
    will have to find.
    """
    asked = set(normalize(query).split())
    extras = []
    for item in result.ner:
        tokens = set(normalize(item).split())
        if tokens and not tokens & asked:
            extras.append(item)
        if len(extras) == limit:
            break
    return extras


def describe_recipe(result: SearchResult, query: str) -> str:
    """One line a human would actually say about a recipe."""
    parts = []
    extras = extra_ingredients(result, query)
    if extras:
        parts.append(f"also needs {join_naturally(extras)}")
    if result.directions:
        steps = len(result.directions)
        parts.append(f"{steps} step{'s' if steps != 1 else ''}")
    return "; ".join(parts) if parts else "no other details recorded"


def format_results(results: list[SearchResult], query: str,
                   offset: int = 0) -> str:
    """Phrase a result set the way a person would.

    Still template-generated -- there is no model here -- but the shape
    is chosen to answer the question a reader actually has ("can I make
    this, and what else do I need") rather than to display the data
    structure. The lead line reflects how good the matches really are,
    because opening with "Here are 5 recipes" when the best scores 12%
    is a small lie told five times a session.
    """
    if not results:
        return NO_RESULTS

    best = results[0].score
    # Prose-join only short ingredient-style queries. Longer ones read
    # badly as "a, b, c and d" and are clearer quoted verbatim.
    words = query.split()
    subject = (join_naturally(words) if 0 < len(words) <= 3
               else f'"{query}"')

    if offset:
        lead = f"A few more using {subject}:"
    elif best >= 0.45:
        lead = f"Good match for {subject} — here's what I'd start with:"
    elif best >= 0.25:
        lead = f"Here's what I found for {subject}:"
    else:
        lead = (f"Nothing matches {subject} closely, but these are the "
                "nearest I have:")

    lines = [lead, ""]
    for position, result in enumerate(results, 1):
        lines.append(f"{position}. {result.title}")
        lines.append(f"   {describe_recipe(result, query)}")
    lines.append("")
    lines.append("Say a number for the full recipe, or ask for "
                 "\"more options\".")
    return "\n".join(lines)


NO_PANTRY_ITEMS = (
    "Tell me what you've got and I'll see what you can make — "
    "for example: \"I have chicken, onion and tomato but no cheese\"."
)

NO_PANTRY_MATCHES = (
    "Nothing in the database comes close to that combination. "
    "Try dropping an ingredient, or add a common one like rice or eggs."
)


def format_pantry(matches, have: list[str], exclude: list[str]) -> str:
    """The pantry answer: what you can make, and what you're missing.

    Shows the missing items by name rather than only a score, because
    "87%" does not tell you whether to go to the shop. The checklist is
    the useful part; the percentage is the ordering.
    """
    if not matches:
        return NO_PANTRY_MATCHES

    complete = [m for m in matches if m.is_complete]
    if len(complete) == 1:
        lead = "One of these you can make right now:"
    elif complete:
        lead = f"You can make {len(complete)} of these without shopping:"
    else:
        shortest = min(len(m.missing) for m in matches)
        lead = (f"Nothing's a perfect fit — the closest needs "
                f"{shortest} more thing{'s' if shortest != 1 else ''}:")

    lines = [lead, ""]
    for match in matches:
        lines.append(f"{match.title}   —   {match.match:.0%} match")
        for item in match.have:
            lines.append(f"   [x] {item}")
        for item in match.missing:
            lines.append(f"   [ ] {item}   (missing)")
        if match.assumed_staples:
            lines.append("   assuming you have: "
                         + ", ".join(match.assumed_staples))
        lines.append("")

    if exclude:
        lines.append(f"Excluded: {', '.join(exclude)}.")
    lines.append("Say a number for the full recipe.")
    return "\n".join(lines)


def format_selection(result: SearchResult) -> str:
    """A sentence about the chosen recipe, not an underlined header."""
    ingredients = len(result.ingredients)
    steps = len(result.directions)
    scale = ("a quick one" if steps <= 3 and ingredients <= 6
             else "a bit of a project" if steps >= 8 or ingredients >= 12
             else "straightforward enough")

    return (
        f"{result.title} — {scale}: {ingredients} ingredients, "
        f"{steps} step{'s' if steps != 1 else ''}.\n\n"
        "Want the ingredients or the method?"
    )


def format_ingredients(result: SearchResult) -> str:
    count = len(result.ingredients)
    lines = [f"You'll need {count} thing{'s' if count != 1 else ''} "
             f"for {result.title}:", ""]
    lines += [f"  - {item}" for item in result.ingredients]
    lines.append("")
    lines.append("Ask for the method when you're ready.")
    return "\n".join(lines)


def format_directions(result: SearchResult) -> str:
    lines = [f"{result.title}, step by step:", ""]
    lines += [f"  {n}. {step}" for n, step in enumerate(result.directions, 1)]
    return "\n".join(lines)


def format_link(result: SearchResult) -> str:
    if not result.link:
        return f"I don't have a source link for {result.title}."
    return f"{result.title} comes from: {result.link}"
