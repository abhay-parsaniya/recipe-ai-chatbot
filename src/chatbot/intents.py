"""Rule-based intent detection.

No LLM, no classifier: a small ordered set of patterns. This is a
deliberate choice, not a placeholder. The conversation has perhaps eight
things a user can mean, the vocabulary is tiny, and rules are inspectable
-- when the bot misreads a message you can point at the exact line. When
an LLM is added later it replaces *this file only*; everything downstream
takes an Intent and does not care how it was produced.

Order matters: patterns are tried top to bottom and the first hit wins,
so specific phrasings must precede general ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    GREET = "greet"
    HELP = "help"
    SEARCH = "search"          # "what can I make with chicken and rice"
    PANTRY = "pantry"          # "I have chicken and onion but no cheese"
    SELECT = "select"          # "show me the second one", "3"
    INGREDIENTS = "ingredients"  # "what do I need?"
    DIRECTIONS = "directions"  # "how do I make it?"
    LINK = "link"              # "where is this from?"
    MORE = "more"              # "show me other options"
    RESET = "reset"            # "start over"
    GOODBYE = "goodbye"
    OUT_OF_DOMAIN = "out_of_domain"   # set by guardrails, not by pattern
    UNKNOWN = "unknown"


# Written as (intent, pattern). Checked in order.
PATTERNS: list[tuple[Intent, re.Pattern]] = [
    (Intent.GOODBYE, re.compile(r"^\s*(bye|goodbye|quit|exit|thanks?( you)?)\b")),
    (Intent.RESET, re.compile(r"\b(start over|reset|new search|clear)\b")),
    (Intent.HELP, re.compile(r"^\s*help\b|\b(what can you do|how does this work)\b")),
    (Intent.GREET, re.compile(r"^\s*(hi|hey|hello|good (morning|evening))\b")),
    (Intent.MORE, re.compile(r"\b(more|other|another|different|next)\b.*"
                             r"\b(option|recipe|result|one)s?\b|^\s*more\s*$")),
    # Trailing `s?` throughout: \bstep\b does NOT match "steps", because
    # the boundary falls inside the word.
    (Intent.DIRECTIONS, re.compile(r"\b(how (do|to)|directions?|instructions?|"
                                   r"steps?|method|cook it|make (it|this)|"
                                   r"prepare)\b")),
    (Intent.INGREDIENTS, re.compile(r"\b(ingredients?|what do i need|what's in|"
                                    r"whats in|shopping list)\b")),
    (Intent.LINK, re.compile(r"\b(links?|url|sources?|where .*(from|find))\b")),
    # SELECT is last of the specific intents: "2" or "the second one".
    (Intent.SELECT, re.compile(r"^\s*#?\d+\s*$"                      # "2"
                               r"|\b(show|pick|choose|select|open"       # verbs
                               r"|tell me about)\b"
                               r"|\b(first|second|third|fourth|fifth"    # ordinals
                               r"|last)\b"
                               r"|\bnumber\s+(\d+|two|three|four|five)\b")),
]

# Ordered, not a dict: "the second one" contains "one", so true ordinals
# must be tested before bare cardinals or every selection resolves to 1.
# "one" is omitted entirely -- in this domain it is a pronoun ("that one"),
# never a position.
ORDINAL_WORDS: list[tuple[str, int]] = [
    ("first", 1), ("1st", 1),
    ("second", 2), ("2nd", 2),
    ("third", 3), ("3rd", 3),
    ("fourth", 4), ("4th", 4),
    ("fifth", 5), ("5th", 5),
    ("two", 2), ("three", 3), ("four", 4), ("five", 5),
]

# Phrases that mean "search" outright, even if another pattern might match.
SEARCH_TRIGGERS = re.compile(
    r"\b(what can i (make|cook)|recipe|i have|suggest|recommend|"
    r"looking for|find me|something with)\b"
)


@dataclass
class ParsedMessage:
    """What the bot understood from one user turn."""

    intent: Intent
    text: str
    position: int | None = None      # 1-based, for SELECT
    terms: list[str] = field(default_factory=list)  # for SEARCH
    have: list[str] = field(default_factory=list)      # for PANTRY
    exclude: list[str] = field(default_factory=list)   # for PANTRY


def extract_position(text: str) -> int | None:
    """Pull a 1-based result number out of "the 2nd one" / "#3" / "3"."""
    lowered = text.lower()
    digits = re.search(r"#?(\d+)", lowered)
    if digits:
        return int(digits.group(1))
    if re.search(r"\blast\b", lowered):
        return -1
    for word, value in ORDINAL_WORDS:
        if re.search(rf"\b{word}\b", lowered):
            return value
    return None


def detect_intent(text: str, has_results: bool = False,
                  has_selection: bool = False) -> Intent:
    """Classify one message.

    ``has_results`` / ``has_selection`` are conversation state: the same
    words mean different things at different points.

    A follow-up with nothing to follow up on keeps its own intent rather
    than being rewritten as a search. "how do I make it" is a cooking
    question missing an antecedent, and the handler answers it with
    "tell me some ingredients first". Turning it into a search sent an
    empty, contentless query downstream, where the domain guardrail
    quite reasonably read it as off-topic.
    """
    lowered = text.lower().strip()
    if not lowered:
        return Intent.UNKNOWN

    if SEARCH_TRIGGERS.search(lowered):
        return Intent.SEARCH

    for intent, pattern in PATTERNS:
        if not pattern.search(lowered):
            continue
        return intent

    return Intent.SEARCH if len(lowered.split()) <= 12 else Intent.UNKNOWN
