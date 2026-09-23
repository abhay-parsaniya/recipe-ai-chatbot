"""Ingredient-based recommendation."""

import pandas as pd
import pytest

from src.chatbot import Conversation, Intent, RecipeChatbot
from src.chatbot.pantry_query import (
    looks_like_pantry_request,
    parse_pantry,
    split_items,
)
from src.config.settings import Settings
from src.data.preprocess import preprocess
from src.search import RecipeSearch
from src.search.pantry import PantryMatcher, singularize, tokenize_ingredient


@pytest.fixture
def matcher(tmp_path) -> PantryMatcher:
    raw = pd.DataFrame({
        "title": ["Chicken Tomato Curry", "Cheese Omelette",
                  "Garlic Onion Soup", "Plain Boiled Rice",
                  "Beef Stew"],
        "ingredients": [
            '["1 chicken", "2 tomatoes", "1 onion", "3 cloves garlic", "ginger"]',
            '["3 eggs", "cheddar cheese", "butter"]',
            '["4 onions", "2 cloves garlic", "stock"]',
            '["2 c. rice", "water", "salt"]',
            '["1 lb. beef", "2 carrots", "1 onion", "stock"]',
        ],
        "directions": [
            '["Fry.", "Simmer."]', '["Beat eggs.", "Fry."]',
            '["Caramelise.", "Add stock."]', '["Boil."]',
            '["Brown beef.", "Simmer."]',
        ],
        "NER": [
            '["chicken", "tomatoes", "onion", "garlic", "ginger"]',
            '["eggs", "cheese", "butter"]',
            '["onions", "garlic", "stock"]',
            '["rice", "water", "salt"]',
            '["beef", "carrots", "onion", "stock"]',
        ],
        "link": [f"{c}.com" for c in "abcde"],
        "source": ["Gathered"] * 5,
    })
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0, default_top_k=5,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    engine = RecipeSearch(preprocess(raw), settings).fit()
    return PantryMatcher(engine, settings)


def by_title(matches) -> dict:
    return {m.title: m for m in matches}


# -- token handling -----------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("tomatoes", "tomato"), ("onions", "onion"), ("berries", "berry"),
    ("eggs", "egg"), ("rice", "rice"), ("glass", "glass"),
])
def test_singularize(word, expected):
    assert singularize(word) == expected


def test_tokenize_normalizes_both_sides():
    assert tokenize_ingredient("2 Tomatoes,") >= {"tomato"}


# -- the headline behaviour --------------------------------------------

def test_matches_the_example_case(matcher):
    """have: chicken, onion, tomato, garlic -> curry, missing ginger."""
    matches = by_title(matcher.match(
        have=["chicken", "onion", "tomato", "garlic"], exclude=["cheese"]))

    curry = matches["Chicken Tomato Curry"]
    assert set(curry.have) == {"chicken", "onion", "tomato", "garlic"}
    assert curry.missing == ["ginger"]
    assert curry.match == pytest.approx(4 / 5, abs=1e-4)
    assert not curry.is_complete


def test_scores_recipe_completeness_not_query_overlap(matcher):
    """The metric is have/needed -- the inverse of search coverage."""
    soup = by_title(matcher.match(have=["onion", "garlic"]))["Garlic Onion Soup"]
    assert soup.match == pytest.approx(2 / 3, abs=1e-4)  # stock missing


def test_a_fully_stocked_recipe_is_complete(matcher):
    soup = by_title(matcher.match(
        have=["onion", "garlic", "stock"]))["Garlic Onion Soup"]
    assert soup.is_complete
    assert soup.missing == []
    assert soup.match == 1.0


def test_complete_matches_rank_above_partial_ones(matcher):
    matches = matcher.match(have=["onion", "garlic", "stock", "chicken"])
    assert matches[0].title == "Garlic Onion Soup"
    assert matches[0].is_complete


def test_missing_items_are_named_not_just_counted(matcher):
    """The shopping list is the useful output, not the percentage."""
    stew = by_title(matcher.match(have=["beef", "onion"]))["Beef Stew"]
    assert set(stew.missing) == {"carrots", "stock"}


# -- exclusions ---------------------------------------------------------

def test_excluded_ingredient_removes_the_recipe_entirely(matcher):
    with_cheese = by_title(matcher.match(have=["eggs", "cheese", "butter"]))
    assert "Cheese Omelette" in with_cheese

    without = by_title(matcher.match(have=["eggs", "cheese", "butter"],
                                     exclude=["cheese"]))
    assert "Cheese Omelette" not in without


def test_exclusion_is_not_just_a_ranking_penalty(matcher):
    """A banned ingredient must never appear, at any position."""
    matches = matcher.match(have=["eggs", "butter"], exclude=["cheese"],
                            top_k=50)
    for match in matches:
        assert "cheese" not in " ".join(match.ingredients).lower()


# -- staples ------------------------------------------------------------

def test_staples_are_assumed_by_default(matcher):
    rice = by_title(matcher.match(have=["rice"]))["Plain Boiled Rice"]
    assert rice.is_complete
    assert set(rice.assumed_staples) == {"water", "salt"}


def test_staples_can_be_turned_off(matcher):
    rice = by_title(matcher.match(have=["rice"],
                                  assume_staples=False))["Plain Boiled Rice"]
    assert not rice.is_complete
    assert set(rice.missing) == {"water", "salt"}


# -- edges --------------------------------------------------------------

def test_empty_pantry_returns_nothing(matcher):
    assert matcher.match(have=[]) == []
    assert matcher.match(have=["  "]) == []


def test_unknown_ingredients_return_nothing_rather_than_noise(matcher):
    assert matcher.match(have=["xylophone", "tungsten"]) == []


def test_min_match_filters_weak_results(matcher):
    assert matcher.match(have=["onion"], min_match=0.99) == []


def test_top_k_limits_results(matcher):
    assert len(matcher.match(have=["onion", "garlic"], top_k=1)) == 1


# -- parsing ------------------------------------------------------------

@pytest.mark.parametrize("message,have,exclude", [
    ("I have chicken, onion, tomato and garlic but no cheese",
     ["chicken", "onion", "tomato", "garlic"], ["cheese"]),
    ("I have: chicken, onion. Don't have: cheese",
     ["chicken", "onion"], ["cheese"]),
    ("what can I make with only eggs, flour and milk",
     ["eggs", "flour", "milk"], []),
    ("what can I cook with chicken, potatoes; out of onions",
     ["chicken", "potatoes"], ["onions"]),
])
def test_parse_pantry(message, have, exclude):
    assert parse_pantry(message) == (have, exclude)


def test_quantities_and_units_are_stripped():
    have, _ = parse_pantry("I have 1/2 lb ground beef, 3 cloves garlic, "
                           "2 large onions, 1 cup flour")
    assert have == ["ground beef", "garlic", "onions", "flour"]


def test_an_item_cannot_be_both_owned_and_banned():
    have, exclude = parse_pantry("I have cheese and eggs but no cheese")
    assert "cheese" not in have
    assert "cheese" in exclude


def test_split_items_deduplicates():
    assert split_items("onion, onion, garlic") == ["onion", "garlic"]


@pytest.mark.parametrize("message,is_pantry", [
    # inventory + exclusion
    ("I have chicken, onion and rice but no cheese", True),
    ("what can I cook with chicken, potatoes; out of onions", True),
    # "only" means a closed list
    ("what can I make with only eggs, flour and milk", True),
    # an inventory phrase with enough items
    ("I have eggs, flour and milk", True),
    # NOT pantry: these are ordinary searches
    ("what can I make with chicken, tomato and onion?", False),
    ("I have chicken and rice", False),      # two items, no exclusion
    ("chicken recipes with no dairy", False),  # one ingredient, not a pantry
    ("chicken tomato onion", False),
    ("show me the second one", False),
])
def test_pantry_mode_requires_a_real_inventory(message, is_pantry):
    """The commonest search phrasing must never be hijacked."""
    assert looks_like_pantry_request(message) is is_pantry


# -- chatbot integration ------------------------------------------------

@pytest.fixture
def bot(matcher) -> RecipeChatbot:
    return RecipeChatbot(matcher.engine, matcher.settings)


def test_a_negation_search_is_not_hijacked(bot):
    """One ingredient plus "no X" stays a search with its honest caveat."""
    reply = bot.respond("chicken recipes with no dairy", Conversation())
    assert reply.intent is Intent.SEARCH
    assert "can't reliably exclude" in reply.text


def test_bot_routes_an_inventory_to_the_pantry_handler(bot):
    reply = bot.respond("I have chicken, onion, tomato and garlic "
                        "but no cheese", Conversation())
    assert reply.intent is Intent.PANTRY
    assert reply.results


def test_pantry_reply_shows_the_checklist(bot):
    reply = bot.respond("I have onion, garlic and stock but no cheese",
                        Conversation())
    assert "[x] onion" in reply.text
    assert "missing" in reply.text
    assert "Excluded: cheese" in reply.text


def test_a_plain_ingredient_list_is_still_a_search(bot):
    reply = bot.respond("chicken tomato onion", Conversation())
    assert reply.intent is Intent.SEARCH


def test_follow_ups_work_on_pantry_results(bot):
    chat = Conversation()
    bot.respond("I have onion, garlic and stock", chat)
    selection = bot.respond("1", chat)
    assert selection.intent is Intent.SELECT
    assert selection.selected is not None
    assert bot.respond("how do I make it?", chat).intent is Intent.DIRECTIONS


def test_pantry_respects_the_domain_guardrail(bot):
    reply = bot.respond("what can I make with no political opinions",
                        Conversation())
    assert reply.intent in (Intent.PANTRY, Intent.OUT_OF_DOMAIN)
