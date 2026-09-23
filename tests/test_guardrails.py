"""Domain boundary tests.

Split deliberately: the OUT cases prove the boundary exists, the IN
cases prove it is not a wall. The second set matters more -- a guardrail
that refuses real cooking questions is worse than none.
"""

import pandas as pd
import pytest

from src.chatbot import Conversation, Intent, RecipeChatbot
from src.chatbot.guardrails import OutOfDomain, classify, has_food_signal
from src.chatbot.responses import OUT_OF_DOMAIN, format_out_of_domain
from src.config.settings import Settings
from src.data.preprocess import preprocess
from src.search import RecipeSearch


# -- out of domain ------------------------------------------------------

@pytest.mark.parametrize("message,category", [
    ("Who is the president of India?", OutOfDomain.POLITICS),
    ("when is the next election", OutOfDomain.POLITICS),
    ("What's the weather tomorrow?", OutOfDomain.WEATHER),
    ("is it going to rain later", OutOfDomain.WEATHER),
    ("Tell me a joke.", OutOfDomain.ENTERTAINMENT),
    ("sing me a song", OutOfDomain.ENTERTAINMENT),
    ("write me a python function to sort a list", OutOfDomain.CODING),
    ("what does this stack trace mean", OutOfDomain.CODING),
    ("I have a headache, what should I take?", OutOfDomain.MEDICAL),
    ("what are the symptoms of flu", OutOfDomain.MEDICAL),
    ("what is the capital of France", OutOfDomain.GENERAL_KNOWLEDGE),
    ("who was Napoleon", OutOfDomain.GENERAL_KNOWLEDGE),
    ("are you a real person", OutOfDomain.PERSONAL),
])
def test_off_topic_messages_are_rejected(message, category):
    verdict = classify(message)
    assert not verdict
    assert verdict.category is category


@pytest.mark.parametrize("message", [
    "what's the stock price of Apple",   # "stock" and "apple" are food words
    "who is the president of Ireland",
    "what is the capital of Turkey",     # "turkey" is a food word
])
def test_hard_overrides_beat_an_incidental_food_word(message):
    """These contain food vocabulary but are plainly not cooking questions."""
    assert not classify(message)


def test_unrecognised_questions_fail_open():
    """No catch-all: an unknown question is allowed through to search.

    An earlier version refused any 3+ word question with no lexicon hit,
    which rejected 11 of 40 real cooking questions. A hand-written
    lexicon always has holes, so the holes must fail open -- a stray
    question reaching search gets "I couldn't find anything matching
    that", which is an acceptable answer.
    """
    assert classify("how many kilometres to the moon")


# -- in domain: the cases a careless filter would break -----------------

@pytest.mark.parametrize("message", [
    "what can I make with chicken and rice",
    "how do I cook rice",                      # "how do i" is not a refusal
    "I have 3 eggs and some flour",            # ingredients, no cooking verb
    "chicken tomato onion",                    # bare ingredient list
    "substitute for buttermilk",
    "what should I make for dinner",
    "how long do I bake a potato",
    "something warming for a cold evening",
    "give me a dessert idea",
    "what goes with salmon",
])
def test_genuine_cooking_questions_are_allowed(message):
    assert classify(message)


@pytest.mark.parametrize("message", [
    "what can I cook that's good for a cold",   # mentions illness
    "is chicken safe to eat while pregnant",    # mentions safety
    "a party dish for election night",          # mentions politics
])
def test_food_signal_outweighs_an_off_topic_word(message):
    """These are cooking questions wearing another topic's vocabulary."""
    assert classify(message)


@pytest.mark.parametrize("message", ["2", "bye", "help", "hi", "more",
                                     "the second one", ""])
def test_conversation_control_is_never_rejected(message):
    assert classify(message)


def test_has_food_signal_detects_both_verbs_and_ingredients():
    assert has_food_signal("how do I bake this")
    assert has_food_signal("I have salmon")
    assert not has_food_signal("who won the match")


# -- the refusal message ------------------------------------------------

def test_refusal_states_role_and_capability():
    assert OUT_OF_DOMAIN == (
        "I'm a recipe assistant. I can help you find recipes, ingredients, "
        "cooking instructions, and meal ideas."
    )


def test_every_category_has_a_nudge():
    for category in OutOfDomain:
        text = format_out_of_domain(category.value)
        assert text.startswith(OUT_OF_DOMAIN)
        assert len(text) > len(OUT_OF_DOMAIN)


def test_refusal_does_not_attempt_the_question():
    """No hedged partial answer, no apology spiral."""
    text = format_out_of_domain("politics")
    for word in ("president", "sorry", "I think", "probably"):
        assert word.lower() not in text.lower()


# -- integration --------------------------------------------------------

@pytest.fixture
def bot(tmp_path) -> RecipeChatbot:
    raw = pd.DataFrame({
        "title": ["Garlic Chicken Rice", "Tomato Chicken Stew"],
        "ingredients": ['["2 chicken breasts", "1 c. rice"]',
                        '["2 chicken thighs", "4 tomatoes"]'],
        "directions": ['["Cook rice.", "Fry chicken."]',
                       '["Brown chicken.", "Add tomatoes."]'],
        "NER": ['["chicken", "rice"]', '["chicken", "tomatoes"]'],
        "link": ["a.com", "b.com"],
        "source": ["Gathered"] * 2,
    })
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0, default_top_k=2,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    return RecipeChatbot(RecipeSearch(preprocess(raw), settings).fit(), settings)


def test_bot_refuses_and_returns_no_recipes(bot):
    reply = bot.respond("Who is the president of India?", Conversation())
    assert reply.intent is Intent.OUT_OF_DOMAIN
    assert reply.text.startswith(OUT_OF_DOMAIN)
    assert reply.results == []
    assert reply.selected is None


def test_refusal_never_invents_a_recipe(bot):
    """The old behaviour retrieved 'President Harding's Pudding Pie'."""
    reply = bot.respond("Who is the president of India?", Conversation())
    assert "President" not in reply.text
    assert reply.results == []


def test_guardrail_still_applies_mid_conversation(bot):
    chat = Conversation()
    bot.respond("chicken and rice", chat)
    reply = bot.respond("tell me a joke", chat)
    assert reply.intent is Intent.OUT_OF_DOMAIN


def test_follow_ups_survive_the_guardrail(bot):
    chat = Conversation()
    bot.respond("chicken and rice", chat)
    assert bot.respond("2", chat).intent is Intent.SELECT
    assert bot.respond("how do I make it?", chat).intent is Intent.DIRECTIONS
    assert bot.respond("more options", chat).intent is Intent.MORE


def test_refusal_does_not_call_the_llm(bot):
    """Nothing was retrieved, so there is nothing to ground against."""
    from src.llm import ResponseGenerator

    class Recorder(ResponseGenerator):
        def __init__(self):
            self.called = False

        @property
        def enabled(self):
            return True

        def generate_search_reply(self, *a, **k):
            self.called = True
            raise AssertionError("LLM must not be called for a refusal")

        def generate_followup_reply(self, *a, **k):
            self.called = True
            raise AssertionError("LLM must not be called for a refusal")

    recorder = Recorder()
    bot.generator = recorder
    bot.respond("who is the president of India?", Conversation())
    assert recorder.called is False


def test_refusal_is_recorded_in_the_transcript(bot):
    chat = Conversation()
    bot.respond("tell me a joke", chat)
    assert [m.role for m in chat.transcript()] == ["user", "bot"]


def test_llm_prompt_states_the_domain_boundary():
    from src.llm.prompts import SYSTEM_PROMPT

    assert "## Your domain" in SYSTEM_PROMPT
    assert OUT_OF_DOMAIN in SYSTEM_PROMPT
    assert "medical" in SYSTEM_PROMPT.lower()


# -- accuracy, measured ------------------------------------------------

REAL_COOKING_QUESTIONS = [
    "what's a good side for steak", "any ideas for a quick lunch",
    "how long should I boil an egg", "what temperature for a roast",
    "can I freeze this", "how many does this serve",
    "what's a vegan option", "something for a birthday",
    "I want comfort food", "what's easy to make",
    "give me something spicy", "what do I do with leftover rice",
    "is there a gluten free version", "what can I bring to a potluck",
    "how do I thicken a sauce", "what's in season right now",
    "what can I make with chicken", "chicken tomato onion",
    "I have eggs and flour", "substitute for buttermilk",
    "how do I cook rice", "what goes with salmon",
    "something warming for a cold evening", "a dessert idea",
    "what should I make for dinner", "how long do I bake a potato",
    "is chicken safe to eat while pregnant",
    "what can I cook that's good for a cold",
    "a party dish for election night", "low carb options",
    "how do I make it crispy", "what pairs with red wine",
    "quick weeknight meal", "something to use up stale bread",
    "how do I reheat this", "can I make this ahead",
    "what's a good marinade", "dinner for six people",
    "kid friendly recipes", "anything with under five ingredients",
]

CLEARLY_OFF_TOPIC = [
    "Who is the president of India?", "when is the next election",
    "What's the weather tomorrow?", "is it going to rain later",
    "Tell me a joke.", "sing me a song",
    "write me a python function to sort a list",
    "what does this stack trace mean",
    "I have a headache, what should I take?",
    "what are the symptoms of flu",
    "what is the capital of France", "who was Napoleon",
    "are you a real person", "what's the stock price of Apple",
    "translate this to Spanish", "who won the match last night",
    "what's the meaning of life", "debug my javascript",
    "should I take antibiotics", "how tall is Everest",
]


@pytest.mark.parametrize("message", REAL_COOKING_QUESTIONS)
def test_real_cooking_questions_are_never_refused(message):
    """The expensive failure. Every one of these must get through."""
    assert classify(message), f"refused a real cooking question: {message}"


@pytest.mark.parametrize("message", CLEARLY_OFF_TOPIC)
def test_clearly_off_topic_messages_are_refused(message):
    assert not classify(message), f"let an off-topic message through: {message}"
