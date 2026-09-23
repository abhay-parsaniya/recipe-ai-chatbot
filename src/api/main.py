"""FastAPI application.

Run:  uvicorn src.api.main:app --reload

The index is loaded once in the lifespan handler and shared by every
request. A startup failure is recorded rather than raised, so the
service still boots and /health can report *why* it is degraded --
a container that refuses to start tells an operator much less.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, health, pantry, recipes, search
from src.api.sessions import SessionStore
from src.chatbot import RecipeChatbot
from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

DESCRIPTION = """\
Recipe recommendation over a local RecipeNLG index.

Retrieval is TF-IDF + cosine similarity. It is lexical: "aubergine" will
not match "eggplant", and negations like "no dairy" are not honoured.
Scores are word-overlap, not quality ratings.

* `POST /search` - ranked recipes, no conversation.
* `POST /pantry` - what you can cook from a list of ingredients.
* `POST /chat` - conversational turn with follow-up support.
* `GET /recipes/{id}` - one recipe by id.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.sessions = SessionStore()
    app.state.bot = None
    app.state.startup_error = None

    try:
        app.state.bot = RecipeChatbot.from_index(settings)
        logger.info("API ready: %d recipes, LLM %s",
                    len(app.state.bot.engine.recipes),
                    "on" if app.state.bot.llm_enabled else "off")
    except FileNotFoundError as exc:
        app.state.startup_error = (
            f"{exc} Build it with: python -m scripts.build_dataset "
            "&& python -m scripts.build_index"
        )
        logger.error("Index unavailable: %s", exc)
    except Exception as exc:  # noqa: BLE001
        app.state.startup_error = f"Failed to load the index: {exc}"
        logger.exception("Startup failed")

    yield


app = FastAPI(
    title="Recipe Assistant API",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)

# Wide open because this is a local prototype with no auth and no user
# data. Lock `allow_origins` to the real front end before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(pantry.router)
app.include_router(chat.router)
app.include_router(recipes.router)
