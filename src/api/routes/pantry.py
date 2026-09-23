"""Ingredient-based recommendation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_bot
from src.api.schemas import PantryMatchOut, PantryRequest, PantryResponse
from src.chatbot import RecipeChatbot

router = APIRouter(tags=["pantry"])


@router.post("/pantry", response_model=PantryResponse)
def pantry(payload: PantryRequest,
           bot: RecipeChatbot = Depends(get_bot)) -> PantryResponse:
    """Recipes ranked by how much of them you already have.

    Different question from /search: that one asks how well a recipe
    matches your words, this one asks whether you can actually cook it
    tonight. The `missing` list is the point -- it is the shopping list.
    """
    matches = bot.pantry.match(
        have=payload.have,
        exclude=payload.exclude,
        top_k=payload.top_k,
        min_match=payload.min_match,
        assume_staples=payload.assume_staples,
    )
    return PantryResponse(
        have=payload.have,
        exclude=payload.exclude,
        count=len(matches),
        complete_count=sum(1 for m in matches if m.is_complete),
        matches=[PantryMatchOut(**m.to_dict()) for m in matches],
    )
