"""Retrieval endpoint: no conversation, no LLM, no state."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_bot
from src.api.schemas import RecipeOut, SearchRequest, SearchResponse
from src.chatbot import RecipeChatbot
from src.chatbot.query import build_query

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest,
           bot: RecipeChatbot = Depends(get_bot)) -> SearchResponse:
    """Rank recipes against a query.

    Exposed separately from /chat so a caller that wants ranked recipes
    -- another service, a notebook, a different UI -- does not have to
    accept a conversational wrapper around them.
    """
    query = build_query(payload.query)
    results = bot.engine.search(query, top_k=payload.top_k,
                                min_score=payload.min_score)
    return SearchResponse(
        query=query,
        count=len(results),
        results=[RecipeOut.from_result(r) for r in results],
    )
