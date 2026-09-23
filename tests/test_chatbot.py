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
    assert "Here are" in reply.text
    assert "1. " in reply.text
    assert "match" in reply.text


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
