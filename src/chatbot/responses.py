"""Turn structured results into text.

Kept apart from the bot's logic so the wording can change without
touching behaviour, and so tests can assert on decisions rather than on
prose. When an LLM is added it replaces the functions in this file.
"""

from __future__ import annotations

from src.search import SearchResult

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


def format_results(results: list[SearchResult], query: str,
                   offset: int = 0) -> str:
    """The result list. Shows the match score, because hiding it would
    imply more confidence than a lexical match earns."""
    if not results:
        return NO_RESULTS

    lead = (f"Here are {len(results)} recipes for \"{query}\":"
            if offset == 0 else
            f"Here are {len(results)} more for \"{query}\":")
    lines = [lead, ""]
    for position, result in enumerate(results, 1):
        preview = ", ".join(result.ner[:5]) or "-"
        lines.append(f"{position}. {result.title}  ({result.score:.0%} match)")
        lines.append(f"   {preview}")
    lines.append("")
    lines.append("Say a number to see one in full, or \"more options\".")
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
    lead = (f"You can make {len(complete)} of these right now:"
            if complete else "Nothing is a perfect fit, but these are close:")

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
    lines.append("Say a number to see one in full.")
    return "\n".join(lines)


def format_selection(result: SearchResult) -> str:
    return (
        f"{result.title}\n"
        f"{'-' * len(result.title)}\n"
        f"{len(result.ingredients)} ingredients, "
        f"{len(result.directions)} steps.\n\n"
        "Ask for the ingredients or the directions."
    )


def format_ingredients(result: SearchResult) -> str:
    lines = [f"Ingredients for {result.title}:", ""]
    lines += [f"  - {item}" for item in result.ingredients]
    return "\n".join(lines)


def format_directions(result: SearchResult) -> str:
    lines = [f"How to make {result.title}:", ""]
    lines += [f"  {n}. {step}" for n, step in enumerate(result.directions, 1)]
    return "\n".join(lines)


def format_link(result: SearchResult) -> str:
    if not result.link:
        return f"I don't have a source link for {result.title}."
    return f"{result.title} comes from: {result.link}"
