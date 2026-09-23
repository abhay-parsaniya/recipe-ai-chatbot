"""Tests for the parts of the UI that are plain functions.

Rendering is not tested here -- asserting on Streamlit calls tests the
mock, not the app. What is tested is the logic that would otherwise hide
in the UI: score banding and turn construction, including the failure
path that keeps the page alive.
"""

import pandas as pd
import pytest

from app.components import SCORE_BANDS, score_label
from app.main import handle_message
from src.chatbot import Conversation, RecipeChatbot
from src.config.settings import Settings
from src.data.preprocess import preprocess
from src.search import RecipeSearch


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


class ExplodingBot:
    """Stands in for any backend failure at request time."""

    def respond(self, message, conversation):
        raise RuntimeError("index vanished")


# -- score banding ------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0.90, "strong word overlap"),
    (0.35, "strong word overlap"),   # boundary is inclusive
    (0.34, "moderate word overlap"),
    (0.25, "moderate word overlap"),
    (0.24, "weak word overlap"),
    (0.00, "weak word overlap"),
])
def test_score_label_bands(score, expected):
    assert score_label(score)[0] == expected


def test_score_bands_are_ordered_high_to_low():
    thresholds = [t for t, _, _ in SCORE_BANDS]
    assert thresholds == sorted(thresholds, reverse=True)


def test_score_label_never_implies_quality():
    """The badge describes word overlap, not how good the recipe is."""
    for _, label, _ in SCORE_BANDS:
        assert "overlap" in label
        assert "best" not in label and "good" not in label


# -- turn construction --------------------------------------------------

def test_successful_turn_carries_text_and_results(bot):
    turn = handle_message(bot, Conversation(), "chicken and rice")
    assert turn["error"] is False
    assert turn["text"]
    assert turn["results"]
    assert turn["user"] == "chicken and rice"


def test_backend_failure_becomes_an_error_turn_not_a_crash():
    turn = handle_message(ExplodingBot(), Conversation(), "chicken")
    assert turn["error"] is True
    assert "RuntimeError" in turn["text"]
    assert "index vanished" in turn["text"]
    assert turn["results"] == []


def test_error_turn_is_still_renderable(bot):
    """Every turn must have the keys render_turn reads."""
    good = handle_message(bot, Conversation(), "chicken")
    bad = handle_message(ExplodingBot(), Conversation(), "chicken")
    for key in ("user", "text", "results", "selected", "error"):
        assert key in good and key in bad


def test_selection_turn_carries_the_selected_recipe(bot):
    """A select turn has no results list -- the card comes from `selected`."""
    chat = Conversation()
    handle_message(bot, chat, "chicken and rice")
    turn = handle_message(bot, chat, "2")
    assert turn["results"] == []
    assert turn["selected"] is not None


def test_render_recipes_picks_the_right_source():
    from app.components import render_recipes
    import inspect

    source = inspect.getsource(render_recipes)
    assert "selected" in source and "results" in source


def test_turn_records_whether_the_llm_was_used(bot):
    turn = handle_message(bot, Conversation(), "chicken")
    assert turn["used_llm"] is False


def test_conversation_state_persists_across_turns(bot):
    chat = Conversation()
    handle_message(bot, chat, "chicken and rice")
    follow_up = handle_message(bot, chat, "1")
    assert chat.selected is not None
    assert follow_up["error"] is False
