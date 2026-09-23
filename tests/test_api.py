"""API tests against a tiny in-memory index -- no artifacts, no network."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.sessions import SessionStore
from src.chatbot import RecipeChatbot
from src.config.settings import Settings
from src.data.preprocess import preprocess
from src.search import RecipeSearch


def build_bot(tmp_path) -> RecipeChatbot:
    raw = pd.DataFrame({
        "title": ["Garlic Chicken Rice", "Tomato Chicken Stew", "Onion Soup"],
        "ingredients": ['["2 chicken breasts", "1 c. rice"]',
                        '["2 chicken thighs", "4 tomatoes", "1 onion"]',
                        '["6 onions", "beef stock"]'],
        "directions": ['["Cook rice.", "Fry chicken."]',
                       '["Brown chicken.", "Add tomatoes."]',
                       '["Caramelise onions."]'],
        "NER": ['["chicken", "rice"]', '["chicken", "tomatoes", "onion"]',
                '["onions", "beef stock"]'],
        "link": ["a.com", "b.com", "c.com"],
        "source": ["Gathered"] * 3,
    })
    settings = Settings(tfidf_min_df=1, tfidf_max_df=1.0, default_top_k=2,
                        artifacts_dir=tmp_path, processed_data_dir=tmp_path)
    return RecipeChatbot(RecipeSearch(preprocess(raw), settings).fit(), settings)


@pytest.fixture
def client(tmp_path):
    """Replace the loaded index after startup, so tests never touch disk."""
    with TestClient(app) as test_client:
        app.state.bot = build_bot(tmp_path)
        app.state.startup_error = None
        app.state.sessions = SessionStore()
        yield test_client


@pytest.fixture
def broken_client():
    with TestClient(app) as test_client:
        app.state.bot = None
        app.state.startup_error = "Index missing. Run scripts.build_index"
        app.state.sessions = SessionStore()
        yield test_client


# -- health -------------------------------------------------------------

def test_health_reports_what_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["recipes_indexed"] == 3
    assert body["llm_enabled"] is False


def test_health_is_200_and_degraded_when_the_index_is_missing(broken_client):
    response = broken_client.get("/health")
    assert response.status_code == 200          # not 500: restarting won't help
    assert response.json()["status"] == "degraded"


# -- search -------------------------------------------------------------

def test_search_returns_ranked_recipes(client):
    body = client.post("/search", json={"query": "chicken tomato onion"}).json()
    assert body["count"] >= 1
    assert body["results"][0]["title"] == "Tomato Chicken Stew"
    assert body["results"][0]["score"] > 0


def test_search_strips_conversational_filler(client):
    body = client.post("/search",
                       json={"query": "hey what can I make with chicken?"}).json()
    assert body["query"] == "chicken"


def test_search_respects_top_k(client):
    body = client.post("/search", json={"query": "chicken", "top_k": 1}).json()
    assert len(body["results"]) == 1


def test_search_min_score_filters(client):
    body = client.post("/search",
                       json={"query": "chicken", "min_score": 0.99}).json()
    assert body["results"] == []


def test_unknown_query_returns_empty_not_an_error(client):
    response = client.post("/search", json={"query": "xylophone tungsten"})
    assert response.status_code == 200
    assert response.json()["count"] == 0


@pytest.mark.parametrize("payload", [
    {},                                  # missing query
    {"query": ""},                       # empty query
    {"query": "chicken", "top_k": 0},    # below minimum
    {"query": "chicken", "top_k": 999},  # above maximum
    {"query": "chicken", "min_score": 2.0},
])
def test_invalid_search_payloads_are_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_search_is_503_when_the_index_is_missing(broken_client):
    response = broken_client.post("/search", json={"query": "chicken"})
    assert response.status_code == 503
    assert "build_index" in response.json()["detail"]


# -- pantry -------------------------------------------------------------

def test_pantry_returns_matches_with_missing_items(client):
    body = client.post("/pantry", json={
        "have": ["chicken", "rice"], "exclude": []}).json()
    assert body["count"] >= 1
    top = body["matches"][0]
    assert 0 < top["match"] <= 1
    assert isinstance(top["missing"], list)
    assert set(top) >= {"recipe_id", "title", "match", "is_complete",
                        "have", "missing", "assumed_staples"}


def test_pantry_exclusion_removes_recipes(client):
    without = client.post("/pantry", json={
        "have": ["chicken", "tomatoes"], "exclude": ["tomatoes"]}).json()
    for match in without["matches"]:
        assert "tomato" not in " ".join(match["ingredients"]).lower()


def test_pantry_reports_how_many_are_fully_cookable(client):
    body = client.post("/pantry", json={"have": ["chicken", "rice"]}).json()
    assert body["complete_count"] == sum(
        1 for m in body["matches"] if m["is_complete"])


def test_pantry_staples_toggle_changes_completeness(client):
    on = client.post("/pantry", json={"have": ["chicken", "rice"],
                                      "assume_staples": True}).json()
    off = client.post("/pantry", json={"have": ["chicken", "rice"],
                                       "assume_staples": False}).json()
    assert on["complete_count"] >= off["complete_count"]


@pytest.mark.parametrize("payload", [
    {}, {"have": []}, {"have": ["chicken"], "top_k": 0},
    {"have": ["chicken"], "min_match": 2.0},
])
def test_invalid_pantry_payloads_are_422(client, payload):
    assert client.post("/pantry", json=payload).status_code == 422


def test_pantry_is_503_when_the_index_is_missing(broken_client):
    assert broken_client.post("/pantry",
                              json={"have": ["chicken"]}).status_code == 503


# -- chat ---------------------------------------------------------------

def test_chat_starts_a_session_and_returns_recipes(client):
    body = client.post("/chat",
                       json={"message": "what can I make with chicken?"}).json()
    assert body["session_id"]
    assert body["intent"] == "search"
    assert body["results"]
    assert body["used_llm"] is False


def test_chat_keeps_context_across_turns(client):
    first = client.post("/chat", json={"message": "chicken tomato onion"}).json()
    session = first["session_id"]

    second = client.post("/chat", json={"message": "show me the second one",
                                        "session_id": session}).json()
    assert second["intent"] == "select"
    assert second["selected"] is not None

    third = client.post("/chat", json={"message": "how do I make it?",
                                       "session_id": session}).json()
    assert third["intent"] == "directions"
    assert third["selected"]["title"] == second["selected"]["title"]


def test_sessions_are_isolated(client):
    a = client.post("/chat", json={"message": "chicken"}).json()
    b = client.post("/chat", json={"message": "onion soup"}).json()
    assert a["session_id"] != b["session_id"]
    assert a["results"][0]["title"] != b["results"][0]["title"]


def test_unknown_session_id_starts_a_fresh_one(client):
    body = client.post("/chat", json={"message": "chicken",
                                      "session_id": "does-not-exist"}).json()
    assert body["session_id"] != "does-not-exist"
    assert body["results"]


def test_client_cannot_choose_its_own_session_id(client):
    body = client.post("/chat", json={"message": "chicken",
                                      "session_id": "attacker-chosen"}).json()
    assert body["session_id"] != "attacker-chosen"


@pytest.mark.parametrize("payload", [{}, {"message": ""},
                                     {"message": "x" * 2001}])
def test_invalid_chat_payloads_are_422(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_is_503_when_the_index_is_missing(broken_client):
    assert broken_client.post("/chat",
                              json={"message": "hi"}).status_code == 503


# -- history / session lifecycle ---------------------------------------

def test_history_returns_both_sides(client):
    session = client.post("/chat", json={"message": "chicken"}).json()["session_id"]
    body = client.get(f"/chat/{session}/history").json()
    assert [m["role"] for m in body["messages"]] == ["user", "bot"]
    assert body["messages"][0]["text"] == "chicken"


def test_history_includes_the_selection(client):
    session = client.post("/chat", json={"message": "chicken"}).json()["session_id"]
    client.post("/chat", json={"message": "1", "session_id": session})
    assert client.get(f"/chat/{session}/history").json()["selected"] is not None


def test_history_of_an_unknown_session_is_404(client):
    assert client.get("/chat/nope/history").status_code == 404


def test_deleting_a_session_removes_it(client):
    session = client.post("/chat", json={"message": "chicken"}).json()["session_id"]
    assert client.delete(f"/chat/{session}").status_code == 204
    assert client.get(f"/chat/{session}/history").status_code == 404


def test_deleting_an_unknown_session_is_404(client):
    assert client.delete("/chat/nope").status_code == 404


# -- recipe lookup ------------------------------------------------------

def test_get_recipe_by_id(client):
    body = client.get("/recipes/0").json()
    assert body["recipe_id"] == 0
    assert body["ingredients"]
    assert body["directions"]


def test_missing_recipe_is_404(client):
    assert client.get("/recipes/9999").status_code == 404


def test_negative_recipe_id_is_422(client):
    assert client.get("/recipes/-1").status_code == 422


# -- contract -----------------------------------------------------------

def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/search" in schema["paths"]
    assert "/chat" in schema["paths"]


def test_recipe_payload_shape_is_stable(client):
    recipe = client.post("/search", json={"query": "chicken"}).json()["results"][0]
    assert set(recipe) == {"recipe_id", "title", "score", "similarity",
                           "coverage", "ingredients", "directions", "ner",
                           "link", "matched_terms"}


def test_search_exposes_the_coverage_breakdown(client):
    results = client.post("/search",
                          json={"query": "chicken tomato onion"}).json()["results"]
    top = results[0]
    assert top["coverage"] == 1.0
    assert top["score"] <= top["similarity"]
