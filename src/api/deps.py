"""Shared dependencies.

The index is loaded once at startup and injected, never built per
request -- fitting TF-IDF on 46k recipes takes seconds, and doing it
inside a handler would make every call time out.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from src.chatbot import RecipeChatbot
from src.api.sessions import SessionStore


def get_bot(request: Request) -> RecipeChatbot:
    """The process-wide chatbot, or a clear 503 if startup failed.

    Returning 503 rather than 500 is the honest code: the service is
    correctly deployed but its data is missing, and a retry after the
    index is built will succeed.
    """
    bot = getattr(request.app.state, "bot", None)
    if bot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "startup_error",
                           "Recipe index unavailable."),
        )
    return bot


def get_sessions(request: Request) -> SessionStore:
    return request.app.state.sessions
