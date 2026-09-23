"""Conversation memory.

Two distinct things are remembered, and conflating them is the usual bug:

* **Transcript** -- every turn, append-only. Used for display and, later,
  as LLM context.
* **State** -- what the pronouns currently refer to: the results on
  screen and which one is selected. "how do I make it" is unanswerable
  without this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.search import SearchResult


@dataclass
class Message:
    role: str            # "user" | "bot"
    text: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Conversation:
    """One session. Cheap to construct, so the API can keep one per user."""

    messages: list[Message] = field(default_factory=list)
    last_results: list[SearchResult] = field(default_factory=list)
    selected: SearchResult | None = None
    last_query: str = ""
    # Results already shown, so "more options" advances instead of
    # repeating the same five recipes.
    offset: int = 0

    # -- transcript -----------------------------------------------------

    def add_user(self, text: str) -> Message:
        return self._add("user", text)

    def add_bot(self, text: str) -> Message:
        return self._add("bot", text)

    def _add(self, role: str, text: str) -> Message:
        message = Message(role=role, text=text)
        self.messages.append(message)
        return message

    def transcript(self, limit: int | None = None) -> list[Message]:
        return self.messages[-limit:] if limit else list(self.messages)

    # -- state ----------------------------------------------------------

    def set_results(self, results: list[SearchResult], query: str) -> None:
        self.last_results = results
        self.last_query = query
        self.selected = None
        self.offset = 0

    def extend_results(self, results: list[SearchResult]) -> None:
        self.offset += len(self.last_results)
        self.last_results = results
        self.selected = None

    def select(self, position: int) -> SearchResult | None:
        """Select by 1-based position; -1 means the last shown."""
        if not self.last_results:
            return None
        index = len(self.last_results) - 1 if position == -1 else position - 1
        if not 0 <= index < len(self.last_results):
            return None
        self.selected = self.last_results[index]
        return self.selected

    @property
    def has_results(self) -> bool:
        return bool(self.last_results)

    @property
    def has_selection(self) -> bool:
        return self.selected is not None

    def reset(self) -> None:
        """Clear state but keep the transcript -- the history is the point."""
        self.last_results = []
        self.selected = None
        self.last_query = ""
        self.offset = 0
