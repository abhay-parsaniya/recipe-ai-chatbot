"""Streamlit session wiring.

The only file in `app/` that knows about Streamlit's session lifecycle.
It holds no recipe logic -- it loads the bot from `src` and stores a
Conversation. If this file were deleted, the backend would be unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.chatbot import Conversation, RecipeChatbot
from src.config.settings import get_settings


@dataclass
class LoadFailure:
    """A startup problem, phrased for a human plus the fix."""

    headline: str
    detail: str
    fix: str = ""


@st.cache_resource(show_spinner=False)
def load_bot() -> tuple[RecipeChatbot | None, LoadFailure | None]:
    """Load the index once per server process, not once per rerun.

    Streamlit reruns the whole script on every interaction. Without
    cache_resource the 28 MB index would be unpickled on every keystroke.
    Returns a failure object rather than raising, so `main.py` can render
    a useful page instead of a stack trace.
    """
    settings = get_settings()
    try:
        return RecipeChatbot.from_index(settings), None
    except FileNotFoundError as exc:
        return None, LoadFailure(
            headline="The recipe index hasn't been built yet.",
            detail=str(exc),
            fix="python -m scripts.build_dataset\npython -m scripts.build_index",
        )
    except ValueError as exc:
        # Raised when the index row count and the parquet disagree.
        return None, LoadFailure(
            headline="The index and the recipe data are out of sync.",
            detail=str(exc),
            fix="python -m scripts.build_index",
        )
    except Exception as exc:  # noqa: BLE001
        return None, LoadFailure(
            headline="Couldn't start the recipe engine.",
            detail=f"{type(exc).__name__}: {exc}",
        )


def get_conversation() -> Conversation:
    """One Conversation per browser session."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()
    return st.session_state.conversation


def get_turns() -> list[dict]:
    """Rendered turns.

    Kept separately from `Conversation.messages` on purpose: the
    transcript stores text, but the UI needs the structured results to
    redraw recipe cards on every rerun. Deriving one from the other
    would mean re-running the search.
    """
    if "turns" not in st.session_state:
        st.session_state.turns = []
    return st.session_state.turns


def reset_session() -> None:
    st.session_state.conversation = Conversation()
    st.session_state.turns = []
