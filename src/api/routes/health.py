from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness + what the service actually has loaded.

    Returns 200 even when the index is missing, with status
    "degraded" -- a health check that 500s tells an orchestrator to
    restart the pod, which will not conjure an index.
    """
    bot = getattr(request.app.state, "bot", None)
    if bot is None:
        return HealthResponse(status="degraded", recipes_indexed=0,
                              llm_enabled=False)
    return HealthResponse(
        status="ok",
        recipes_indexed=len(bot.engine.recipes),
        llm_enabled=bot.llm_enabled,
        model=bot.settings.llm_model if bot.llm_enabled else None,
    )
