"""End-to-end smoke test: actually executes app/main.py.

Needs the built index, so it skips when the artifacts are absent rather
than failing on a fresh clone.
"""

import pytest

from src.config.settings import get_settings

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not get_settings().index_path.exists(),
    reason="index not built; run scripts.build_index",
)


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file("app/main.py", default_timeout=300)
    at.run()
    return at


def test_app_starts_without_exceptions(app):
    assert app.exception == []
    assert app.title[0].value.endswith("Recipe Assistant")


def test_search_renders_recipe_cards(app):
    app.chat_input[0].set_value("chicken tomato onion").run()
    assert app.exception == []
    assert len(app.expander) >= 1


def test_follow_up_renders_the_selected_recipe(app):
    app.chat_input[0].set_value("show me the second one").run()
    assert app.exception == []
    # The newest card is the single selected recipe.
    assert app.expander[-1].label.startswith("1. ")


def test_unknown_query_does_not_crash(app):
    app.chat_input[0].set_value("xylophone tungsten").run()
    assert app.exception == []


def test_reset_clears_the_conversation(app):
    app.sidebar.button[0].click().run()
    assert app.exception == []
    assert len(app.chat_message) == 1
