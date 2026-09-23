import pandas as pd
import pytest

from src.chatbot import BotReply, Conversation, Intent, RecipeChatbot
from src.chatbot.query import build_query, extract_terms, has_negation
from src.config.settings import Settings
from src.data.preprocess import preprocess
from src.search import RecipeSearch


@pytest.fixture
def bot(tmp_path) -> RecipeChatbot:
    raw = pd.DataFrame(
        {
            "title": ["Garlic Chicken Rice", "Chocolate Banana Bread",
                      "Beef Chili", "Lemon Garlic Shrimp", "Vanilla Pancakes",
                      "Tomato Chicken Stew", "Onion Soup"],
            "ingredients": [
                '["2 chicken breasts", "1 c. rice", "3 cloves garlic"]',
                '["3 bananas", "2 c. flour", "1/2 c. cocoa"]',
                '["1 lb. ground beef", "2 cans kidney beans"]',
                '["1 lb. shrimp", "4 cloves garlic", "1 lemon"]',
                '["2 c. flour", "1 tsp. vanilla", "2 eggs"]',
                '["2 chicken thighs", "4 tomatoes", "1 onion"]',
                '["6 onions", "beef stock", "2 c. gruyere"]',
            ],
            "directions": [
                '["Cook the rice.", "Fry the chicken."]',
                '["Mash bananas.", "Bake 60 minutes."]',
                '["Brown the beef.", "Simmer."]',
                '["Saute garlic.", "Add shrimp."]',
                '["Whisk batter.", "Fry."]',
                '["Brown the chicken.", "Add tomatoes.", "Simmer 40 min."]',
                '["Caramelise onions.", "Add stock."]',
            ],
            "NER": [
                '["chicken", "rice", "garlic"]',
                '["bananas", "flour", "cocoa"]',
                '["ground beef", "kidney beans"]',
                '["shrimp", "garlic", "lemon"]',
                '["flour", "vanilla", "eggs"]',
                '["chicken", "tomatoes", "onion"]',
                '["onions", "beef stock", "gruyere"]',
            ],
            "link": [f"{c}.com" for c in "abcdefg"],
            "source": ["Gathered"] * 7,
        }
    )
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0, default_top_k=3,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    engine = RecipeSearch(preprocess(raw), settings).fit()
    return RecipeChatbot(engine, settings)


@pytest.fixture
def chat() -> Conversation:
    return Conversation()


# -- query parsing ------------------------------------------------------

def test_extract_terms_strips_conversational_filler():
    terms = extract_terms("hey, what can I make with chicken and rice tonight?")
    assert terms == ["chicken", "rice"]


def test_extract_terms_deduplicates():
    assert extract_terms("chicken chicken rice") == ["chicken", "rice"]


def test_build_query_falls_back_when_everything_is_filler():
    assert build_query("what can I make") != ""


def test_has_negation_detects_exclusions():
    assert has_negation("chicken but no dairy")
    assert not has_negation("chicken and rice")


# -- search turn --------------------------------------------------------

def test_natural_language_query_returns_recipes(bot, chat):
    reply = bot.respond("what can I make with chicken, tomato and onion?", chat)
    assert reply.intent is Intent.SEARCH
    assert reply.results
    assert reply.results[0].title == "Tomato Chicken Stew"


def test_search_respects_top_k(bot, chat):
    reply = bot.respond("I have chicken and rice", chat)
    assert len(reply.results) <= bot.settings.default_top_k


def test_response_is_conversational_not_a_data_dump(bot, chat):
    reply = bot.respond("I have chicken and rice", chat)
    text = reply.text
    assert "1. " in text                      # still selectable by number
    assert "step" in text                     # says how involved each is
    assert text.split("\n")[0].endswith(":")  # opens with a sentence
    # The lead must not be the same regardless of how good the match is.
    assert "Here are 5 recipes for" not in text


def test_lead_line_reflects_match_quality(bot, chat):
    """Opening with "here's what I found" when the best hit scores 12%
    is a small lie told every session."""
    from src.chatbot.responses import format_results

    strong = format_results([_scored(0.60)], "chicken")
    weak = format_results([_scored(0.05)], "chicken")
    assert strong.split("\n")[0] != weak.split("\n")[0]
    assert "closely" in weak


def test_listing_shows_what_else_a_recipe_needs(bot, chat):
    """Repeating the user's own ingredients back is not information."""
    from src.chatbot.responses import extra_ingredients
    from src.search import SearchResult

    result = SearchResult(recipe_id=1, title="X", score=0.5,
                          ner=["chicken", "rice", "saffron", "stock"])
    extras = extra_ingredients(result, "chicken rice")
    assert "saffron" in extras
    assert "chicken" not in extras


def _scored(score: float):
    from src.search import SearchResult

    return SearchResult(recipe_id=1, title="Test Recipe", score=score,
                        ner=["butter"], directions=["Cook."])


def test_negation_is_flagged_not_silently_ignored(bot, chat):
    reply = bot.respond("chicken recipes with no dairy", chat)
    assert "can't reliably exclude" in reply.text


def test_no_results_gets_a_helpful_message(bot, chat):
    reply = bot.respond("xylophone tungsten quarks", chat)
    assert reply.results == []
    assert "couldn't find" in reply.text


# -- follow-ups ---------------------------------------------------------

def test_select_by_number_then_ask_for_directions(bot, chat):
    bot.respond("I have chicken, tomato and onion", chat)
    selection = bot.respond("2", chat)
    assert selection.intent is Intent.SELECT
    assert selection.selected is not None

    steps = bot.respond("how do I make it?", chat)
    assert steps.intent is Intent.DIRECTIONS
    assert steps.selected.title == selection.selected.title
    assert "1." in steps.text


@pytest.mark.parametrize("phrase,index", [
    ("show me the first one", 0),
    # "the second one" contains "one" -- a naive ordinal lookup returns 1.
    ("show me the second one", 1),
    ("the third one", 2),
    ("number two", 1),
    ("the last one", -1),
])
def test_select_by_ordinal_word(bot, chat, phrase, index):
    bot.respond("chicken tomato onion", chat)
    results = list(chat.last_results)
    reply = bot.respond(phrase, chat)
    assert reply.selected.title == results[index].title


def test_ingredients_follow_up(bot, chat):
    bot.respond("chicken tomato onion", chat)
    bot.respond("1", chat)
    reply = bot.respond("what ingredients do I need?", chat)
    assert reply.intent is Intent.INGREDIENTS
    assert all(i in reply.text for i in reply.selected.ingredients)


def test_link_follow_up(bot, chat):
    bot.respond("chicken tomato onion", chat)
    bot.respond("1", chat)
    reply = bot.respond("where is this from?", chat)
    assert reply.selected.link in reply.text


def test_out_of_range_selection_is_rejected_clearly(bot, chat):
    bot.respond("chicken tomato onion", chat)
    reply = bot.respond("99", chat)
    assert reply.selected is None
    assert "pick 1 to" in reply.text


@pytest.mark.parametrize("message,intent", [
    ("how do I make it?", Intent.DIRECTIONS),
    ("what ingredients do I need?", Intent.INGREDIENTS),
    ("where is this from?", Intent.LINK),
    ("show me the second one", Intent.SELECT),
    ("more options", Intent.MORE),
])
def test_follow_up_without_a_search_asks_for_ingredients(bot, chat, message,
                                                         intent):
    """A follow-up with no antecedent keeps its intent and says what's
    missing, rather than being laundered into an empty search."""
    reply = bot.respond(message, chat)
    assert reply.intent is intent
    assert "ingredients first" in reply.text


def test_single_result_is_auto_selected(bot, chat):
    bot.settings.default_top_k = 1
    bot.respond("gruyere", chat)
    reply = bot.respond("how do I make it?", chat)
    assert reply.selected is not None


# -- more options -------------------------------------------------------

def test_more_options_returns_different_recipes(bot, chat):
    first = bot.respond("chicken garlic rice", chat)
    second = bot.respond("show me other options", chat)
    assert second.intent is Intent.MORE
    first_titles = {r.title for r in first.results}
    assert not first_titles & {r.title for r in second.results}


def test_more_options_eventually_runs_out(bot, chat):
    bot.respond("chicken garlic rice", chat)
    for _ in range(6):
        reply = bot.respond("more options", chat)
    assert "everything I have" in reply.text


# -- history ------------------------------------------------------------

def test_transcript_records_both_sides_in_order(bot, chat):
    bot.respond("hi", chat)
    bot.respond("chicken tomato onion", chat)
    roles = [m.role for m in chat.transcript()]
    assert roles == ["user", "bot", "user", "bot"]
    assert chat.messages[0].text == "hi"


def test_transcript_limit_returns_the_tail(bot, chat):
    for _ in range(4):
        bot.respond("chicken", chat)
    assert len(chat.transcript(limit=2)) == 2


def test_reset_clears_state_but_keeps_history(bot, chat):
    bot.respond("chicken tomato onion", chat)
    bot.respond("1", chat)
    bot.respond("start over", chat)
    assert chat.last_results == []
    assert chat.selected is None
    assert len(chat.messages) == 6


def test_conversations_are_independent(bot):
    a, b = Conversation(), Conversation()
    bot.respond("chicken tomato onion", a)
    bot.respond("bananas cocoa", b)
    assert a.last_results[0].title != b.last_results[0].title


# -- small talk ---------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("hi", Intent.GREET),
    ("help", Intent.HELP),
    ("bye", Intent.GOODBYE),
])
def test_small_talk_intents(bot, chat, message, expected):
    assert bot.respond(message, chat).intent is expected


def test_reply_is_json_serializable(bot, chat):
    import json
    payload = bot.respond("chicken tomato onion", chat).to_dict()
    assert json.loads(json.dumps(payload))["intent"] == "search"


def test_contractions_are_stripped_from_queries():
    """Regression: normalize() deletes apostrophes, so "what's" arrives
    as "whats". Without it in STOPWORDS the bot searched for the literal
    word "whats"."""
    assert build_query("what's a good side for steak") == "side steak"
    assert "whats" not in build_query("what's good with salmon")
    assert "dont" not in build_query("I don't have cheese")
    # When stripping removes everything, build_query deliberately falls
    # back to the raw message -- an empty query matches nothing at all,
    # which is a worse answer than a noisy one.
    assert build_query("what's in this") == "whats in this"


def test_subjective_adjectives_are_not_searched():
    """"good" and "easy" match recipes titled "Good Cake" and tell us
    nothing about what the user wants."""
    query = build_query("a good easy quick chicken dinner")
    assert "chicken" in query
    for word in ("good", "easy", "quick"):
        assert word not in query.split()
