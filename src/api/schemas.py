"""Request and response models.

These are the API's contract, kept separate from the internal
dataclasses on purpose. `SearchResult` may gain or lose fields as
retrieval changes; this file decides what clients actually see, so an
internal refactor cannot silently alter the public shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.chatbot import BotReply
from src.search import SearchResult


class RecipeOut(BaseModel):
    recipe_id: int
    title: str
    score: float = Field(description="Ranking score, 0-1: similarity x "
                                     "coverage. Lexical word overlap, "
                                     "not a quality rating.")
    similarity: float = Field(default=0.0,
                              description="Raw cosine similarity, before "
                                          "the coverage boost.")
    coverage: float = Field(default=1.0,
                            description="Fraction of the query's words this "
                                        "recipe contains. 1.0 means it has "
                                        "all of them.")
    ingredients: list[str] = []
    directions: list[str] = []
    ner: list[str] = Field(default=[], description="Bare ingredient names.")
    link: str = ""
    matched_terms: list[str] = Field(
        default=[], description="Query terms this recipe shares -- the "
                                "reason it scored what it did."
    )

    @classmethod
    def from_result(cls, result: SearchResult) -> "RecipeOut":
        return cls(**result.to_dict())


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500,
                       examples=["chicken tomato onion"])
    top_k: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    query: str = Field(description="The cleaned query actually searched.")
    count: int
    results: list[RecipeOut]


class PantryRequest(BaseModel):
    have: list[str] = Field(min_length=1, max_length=50,
                            examples=[["chicken", "onion", "tomato", "garlic"]])
    exclude: list[str] = Field(default=[], max_length=50,
                               examples=[["cheese"]])
    top_k: int = Field(default=5, ge=1, le=50)
    min_match: float = Field(default=0.0, ge=0.0, le=1.0)
    assume_staples: bool = Field(
        default=True,
        description="Treat salt, water, oil etc. as always available.",
    )


class PantryMatchOut(BaseModel):
    recipe_id: int
    title: str
    match: float = Field(description="Fraction of this recipe's ingredients "
                                     "you already have, 0-1.")
    is_complete: bool = Field(description="True when nothing is missing.")
    have: list[str] = []
    missing: list[str] = []
    assumed_staples: list[str] = []
    ingredients: list[str] = []
    directions: list[str] = []
    link: str = ""


class PantryResponse(BaseModel):
    have: list[str]
    exclude: list[str]
    count: int
    complete_count: int = Field(description="How many need nothing else.")
    matches: list[PantryMatchOut]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000,
                         examples=["what can I make with chicken and rice?"])
    session_id: str | None = Field(
        default=None,
        description="Omit to start a new conversation; the response "
                    "returns an id to send back on the next turn.",
    )


class ChatResponse(BaseModel):
    session_id: str
    text: str
    intent: str
    used_llm: bool = Field(description="True if an LLM phrased this reply. "
                                       "Retrieval ran either way.")
    results: list[RecipeOut] = []
    selected: RecipeOut | None = None

    @classmethod
    def from_reply(cls, reply: BotReply, session_id: str) -> "ChatResponse":
        return cls(
            session_id=session_id,
            text=reply.text,
            intent=reply.intent.value,
            used_llm=reply.used_llm,
            results=[RecipeOut.from_result(r) for r in reply.results],
            selected=(RecipeOut.from_result(reply.selected)
                      if reply.selected else None),
        )


class MessageOut(BaseModel):
    role: str
    text: str
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]
    selected: RecipeOut | None = None


class HealthResponse(BaseModel):
    status: str
    recipes_indexed: int
    llm_enabled: bool
    model: str | None = None


class ErrorResponse(BaseModel):
    detail: str
