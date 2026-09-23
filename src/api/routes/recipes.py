"""Direct recipe lookup by id."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status

from src.api.deps import get_bot
from src.api.schemas import RecipeOut
from src.chatbot import RecipeChatbot
from src.search import SearchResult

router = APIRouter(tags=["recipes"])


@router.get("/recipes/{recipe_id}", response_model=RecipeOut)
def get_recipe(recipe_id: int = Path(ge=0),
               bot: RecipeChatbot = Depends(get_bot)) -> RecipeOut:
    """Fetch one recipe.

    Lets a client render a recipe from a stored id without re-running a
    search, which is what any "save this recipe" feature needs.
    """
    frame = bot.engine.recipes
    matches = frame.index[frame["recipe_id"] == recipe_id]
    if len(matches) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail=f"No recipe with id {recipe_id}")

    row = frame.loc[matches[0]]
    as_list = bot.engine.as_list
    result = SearchResult(
        recipe_id=int(row["recipe_id"]),
        title=str(row["title"]),
        score=1.0,               # an exact lookup, not a ranked match
        ingredients=as_list(row.get("ingredients")),
        directions=as_list(row.get("directions")),
        ner=as_list(row.get("ner")),
        link=str(row.get("link") or ""),
    )
    return RecipeOut.from_result(result)
