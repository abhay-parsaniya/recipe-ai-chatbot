"""Conversational endpoint. Wraps RecipeChatbot; adds no recipe logic."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.api.deps import get_bot, get_sessions
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    MessageOut,
    RecipeOut,
)
from src.api.sessions import SessionStore
from src.chatbot import RecipeChatbot
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest,
         bot: RecipeChatbot = Depends(get_bot),
         sessions: SessionStore = Depends(get_sessions)) -> ChatResponse:
    """One conversational turn.

    Send `session_id` back on each turn to keep context; omit it to
    start fresh. An unknown or expired id silently starts a new session
    rather than erroring -- see SessionStore.get_or_create.
    """
    session_id, conversation = sessions.get_or_create(payload.session_id)

    try:
        reply = bot.respond(payload.message, conversation)
    except Exception as exc:  # noqa: BLE001
        # The LLM layer already degrades internally, so reaching here
        # means a genuine server fault. Log it with the session, return
        # a 500 without leaking internals to the client.
        logger.exception("chat turn failed (session=%s)", session_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to handle that message.",
        ) from exc

    return ChatResponse.from_reply(reply, session_id)


@router.get("/chat/{session_id}/history", response_model=HistoryResponse)
def history(session_id: str,
            sessions: SessionStore = Depends(get_sessions)) -> HistoryResponse:
    conversation = sessions.get(session_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="Unknown or expired session.")
    return HistoryResponse(
        session_id=session_id,
        messages=[MessageOut(role=m.role, text=m.text,
                             timestamp=m.timestamp.isoformat())
                  for m in conversation.transcript()],
        selected=(RecipeOut.from_result(conversation.selected)
                  if conversation.selected else None),
    )


@router.delete("/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def end_session(session_id: str,
                sessions: SessionStore = Depends(get_sessions)) -> Response:
    if not sessions.delete(session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="Unknown or expired session.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
