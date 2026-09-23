"""Domain boundary: decide whether a message is a cooking question.

Why this is a separate layer, ahead of intent detection: without it the
bot treats every unrecognised message as a search. "Who is the president
of India?" then retrieves *President Harding's Pudding Pie* and presents
it as an answer -- confidently wrong, which is worse than useless.

The design principle here is **precision over recall**. A false positive
(refusing a real cooking question) is a visible, annoying failure. A
false negative (letting an odd question through to search) degrades to
"I couldn't find anything matching that", which is already an acceptable
answer. So the out-of-domain patterns are specific and narrow, and any
message with a genuine food signal is let through even if it also trips
one of them -- "is chicken safe to eat while pregnant" mentions a
medical topic but is answered by refusing the *medical* part, not by
refusing a chicken question outright.

Rule-based on purpose: the boundary is inspectable, testable and free.
An LLM classifier here would add latency and cost to every turn, and
would itself need guardrails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.utils.text import normalize


class OutOfDomain(str, Enum):
    POLITICS = "politics"
    WEATHER = "weather"
    GENERAL_KNOWLEDGE = "general_knowledge"
    CODING = "coding"
    MEDICAL = "medical"
    ENTERTAINMENT = "entertainment"
    PERSONAL = "personal"
    OTHER = "other"


# Words that make a message a cooking message. Presence of any of these
# outweighs an out-of-domain pattern, because "how do I cook rice" should
# never be refused for containing "how do I".
FOOD_SIGNALS = re.compile(
    r"\b("
    # preparation
    r"recipe|recipes|cook|cooking|cooked|bake|baking|baked|roast|roasted|"
    r"fry|fried|frying|grill|grilled|grilling|boil|boiled|simmer|saute|"
    r"steam|steamed|poach|braise|stir.?fry|reheat|reheating|defrost|"
    r"freeze|frozen|thaw|marinade|marinate|season|seasoned|knead|whisk|"
    r"chop|dice|mince|prep|prepare|"
    # the meal itself
    r"ingredient|ingredients|dish|dishes|meal|meals|menu|course|"
    r"dinner|lunch|breakfast|brunch|supper|snack|dessert|pudding|"
    r"appetizer|appetiser|starter|sides?|side dish|main|entree|"
    r"potluck|barbecue|bbq|picnic|buffet|takeaway|leftovers?|"
    # asking about one
    r"substitute|substitution|swap|replace .* with|instead of|"
    r"serves?|serving|servings|portion|portions|batch|make ahead|"
    r"pairs? with|goes with|goes well|what to (?:make|cook|serve)|"
    # kit
    r"cuisine|kitchen|oven|stove|hob|skillet|saucepan|casserole|"
    r"microwave|fridge|freezer|pantry|"
    # qualities and diets
    r"eat|eating|edible|hungry|cravings?|tasty|delicious|flavou?r|"
    r"crispy|crunchy|creamy|savou?ry|sweet|spicy|bland|stale|"
    r"vegan|vegetarian|pescatarian|gluten|dairy.?free|low.?carb|keto|"
    r"paleo|calorie|calories|nutritious|filling|comfort food|"
    r"kid.?friendly|weeknight|homemade"
    r")\b"
)

# Common ingredients, so "what can I do with 3 eggs" is recognised as
# food even with no cooking verb. Short but high-frequency; the search
# index holds the long tail.
INGREDIENT_SIGNALS = re.compile(
    r"\b("
    r"chicken|beef|pork|lamb|turkey|bacon|ham|sausage|fish|salmon|tuna|"
    r"shrimp|prawn|egg|eggs|cheese|milk|cream|butter|yogurt|"
    r"rice|pasta|noodle|noodles|bread|flour|dough|potato|potatoes|"
    r"tomato|tomatoes|onion|onions|garlic|carrot|pepper|mushroom|"
    r"spinach|broccoli|bean|beans|lentil|chickpea|tofu|"
    r"apple|banana|lemon|lime|orange|berry|berries|chocolate|"
    r"sugar|salt|oil|vinegar|herb|herbs|spice|spices|sauce|soup|salad|"
    r"cake|pie|cookie|cookies|curry|stew|pizza|sandwich|"
    r"steak|chop|chops|roast|mince|casserole|pancake|pancakes|"
    r"wine|beer|stock|broth|gravy|syrup|honey|jam|nuts|seeds|"
    r"corn|peas|cabbage|lettuce|cucumber|courgette|zucchini|"
    r"aubergine|eggplant|pumpkin|squash|leek|celery|ginger|chilli|chili"
    r")\b"
)

# Phrases that are never about cooking even though they contain a food
# word. Checked *before* the food signal, because "stock price" would
# otherwise be rescued by "stock" and "apple" in a share-price question.
HARD_OUT_OF_DOMAIN: list[tuple[OutOfDomain, re.Pattern]] = [
    (OutOfDomain.GENERAL_KNOWLEDGE, re.compile(
        r"\bstock (?:price|market)\b|\bshare price\b|\bexchange rate\b")),
    (OutOfDomain.POLITICS, re.compile(
        r"\b(?:president|prime minister|chancellor) of\b")),
    (OutOfDomain.GENERAL_KNOWLEDGE, re.compile(r"\bcapital of\b")),
]

# Ordered: first match wins, so specific patterns precede general ones.
OUT_OF_DOMAIN_PATTERNS: list[tuple[OutOfDomain, re.Pattern]] = [
    (OutOfDomain.POLITICS, re.compile(
        r"\b(president|prime minister|senator|congress|parliament|"
        r"election|elections|vote|voting|politic\w*|government|"
        r"democrat|republican|party leader|minister of)\b")),
    (OutOfDomain.WEATHER, re.compile(
        r"\b(weather|forecast|rain|raining|snow|snowing|sunny|humidity|"
        r"how (hot|cold) is it|temperature (outside|today|tomorrow))\b")),
    (OutOfDomain.CODING, re.compile(
        r"\b(python|javascript|java|typescript|sql|html|css|regex|"
        r"function|variable|compile|debug|git|api endpoint|"
        r"write .*(code|script|program)|stack trace|syntax error)\b")),
    (OutOfDomain.MEDICAL, re.compile(
        r"\b(diagnos\w*|symptom|symptoms|headache|fever|infection|"
        r"prescription|medication|medicine|antibiotic|dosage|"
        r"should i take|is it safe to (take|use)|"
        r"cure|treat(ment)? for|disease|illness|doctor)\b")),
    (OutOfDomain.ENTERTAINMENT, re.compile(
        r"\b(joke|jokes|funny|riddle|poem|story|sing|song|lyrics|"
        r"movie|film|tv show|game of|play a game)\b")),
    (OutOfDomain.PERSONAL, re.compile(
        r"\b(who are you|are you (a )?(human|robot|ai|bot|real)|"
        r"your (name|creator|opinion|feelings)|do you (love|feel|think))\b")),
    (OutOfDomain.GENERAL_KNOWLEDGE, re.compile(
        r"\b(capital of|population of|who (is|was|were|won)|when (did|was)|"
        r"where is|how far|how tall|translate|meaning of life|"
        r"stock price|bitcoin|news|history of (?!the recipe))\b")),
]

@dataclass
class DomainVerdict:
    in_domain: bool
    category: OutOfDomain | None = None
    matched: str = ""

    def __bool__(self) -> bool:
        return self.in_domain


def has_food_signal(text: str) -> bool:
    normalized = normalize(text)
    return bool(FOOD_SIGNALS.search(normalized)
                or INGREDIENT_SIGNALS.search(normalized))


def classify(text: str) -> DomainVerdict:
    """Decide whether this message is the recipe assistant's business.

    Food signal wins. A message mentioning both food and an off-limits
    topic is treated as a cooking question, because that is what it
    usually is: "what can I cook that's good for a cold" is a recipe
    request with a medical word in it, not a request for medical advice.
    The system prompt handles the residual case by declining the medical
    part while still offering recipes.
    """
    if not text or not text.strip():
        return DomainVerdict(in_domain=True)

    normalized = normalize(text)

    for category, pattern in HARD_OUT_OF_DOMAIN:
        match = pattern.search(normalized)
        if match:
            return DomainVerdict(False, category, match.group(0))

    if has_food_signal(text):
        return DomainVerdict(in_domain=True)

    for category, pattern in OUT_OF_DOMAIN_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return DomainVerdict(False, category, match.group(0))

    # Deliberately no catch-all for "a question with no food words". An
    # earlier version refused any 3+ word question missing a lexicon
    # hit, which rejected 11 of 40 real cooking questions -- "can I
    # freeze this", "what pairs with red wine", "how many does this
    # serve". A hand-written lexicon will always have holes, so the
    # holes must fail open. An off-topic message that slips through
    # reaches search and gets "I couldn't find anything matching that",
    # which is a fine answer; a refused cooking question is not.
    return DomainVerdict(in_domain=True)
