"""Turn a natural-language message into search terms.

The search engine matches ingredient vocabulary. A raw message like
"hey, what can I make with chicken and rice tonight?" is mostly words
that appear in no recipe -- "hey", "what", "tonight" -- and feeding them
straight in adds noise: TF-IDF will happily match "tonight" against any
recipe whose text contains it.

So we strip the conversational scaffolding and keep the food words. This
is lossy on purpose; the engine only needs the nouns.
"""

from __future__ import annotations

import re

from src.utils.text import normalize

# Whole phrases that wrap a request. Removed before tokenizing so that
# multi-word filler ("what can i make with") goes in one piece.
FILLER_PHRASES = [
    "what can i make with", "what can i cook with", "what can i make",
    "what should i cook", "what should i make", "can you suggest",
    "can you recommend", "do you have a recipe for", "give me a recipe for",
    "i want to make", "i want to cook", "i would like", "i'd like",
    "looking for", "find me", "show me", "something with", "recipe for",
    "recipes with", "recipes for", "recipe with", "i have got", "i have",
    "i've got", "ive got", "in my fridge", "in the fridge", "at home",
    "for dinner", "for lunch", "for breakfast", "tonight", "today",
]

# Single words with no discriminative value in a recipe corpus.
STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "but", "by", "can",
    "could", "do", "does", "for", "from", "get", "give", "got", "has",
    "have", "hey", "hi", "hello", "how", "i", "if", "in", "is", "it",
    "just", "like", "make", "me", "my", "need", "of", "on", "only", "or",
    "please", "recipe", "recipes", "should", "some", "something", "thanks",
    "that", "the", "them", "then", "there", "these", "they", "this", "to",
    "use", "using", "want", "was", "we", "what", "when", "where", "which",
    "will", "with", "without", "would", "you", "your",
}

# Words that flip the meaning of what follows. We detect them so the bot
# can *say* it cannot honour them -- silently ignoring "no dairy" and
# then serving a cheese bake is worse than admitting the limit.
NEGATION_WORDS = re.compile(r"\b(no|not|non|without|free|avoid|except|"
                            r"allerg\w*|dairy-free|gluten-free)\b")


def extract_terms(message: str) -> list[str]:
    """Return the food-ish tokens from a message, order preserved."""
    text = normalize(message)
    for phrase in FILLER_PHRASES:
        text = text.replace(normalize(phrase), " ")
    seen: set[str] = set()
    terms: list[str] = []
    for token in text.split():
        if token in STOPWORDS or len(token) < 2 or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def has_negation(message: str) -> bool:
    """True if the user asked to exclude something."""
    return bool(NEGATION_WORDS.search(normalize(message)))


def build_query(message: str) -> str:
    """The string actually handed to RecipeSearch.

    Falls back to the normalized message when stripping removed
    everything -- an empty query would return nothing at all, which is a
    worse answer than a noisy one.
    """
    terms = extract_terms(message)
    return " ".join(terms) if terms else normalize(message)
