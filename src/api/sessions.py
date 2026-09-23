"""In-memory conversation store.

Deliberately simple and deliberately bounded. The chatbot keeps state
per Conversation, and HTTP is stateless, so something has to hold them
between requests.

Two properties that matter more than cleverness here:

* **Bounded.** An unbounded dict keyed on a client-supplied id is a
  memory-exhaustion bug: anyone can mint new session ids in a loop.
  Oldest sessions are evicted past a cap, and idle ones expire.
* **Server-generated ids.** Clients may not choose their own session id,
  so one caller cannot guess or collide with another's conversation.

This is a single-process store. Multiple workers would each hold their
own sessions, so a real deployment needs Redis behind this interface --
which is why the interface is four methods.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from src.chatbot import Conversation


@dataclass
class _Entry:
    conversation: Conversation
    last_used: float


class SessionStore:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: float = 3600):
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._max = max_sessions
        self._ttl = ttl_seconds
        # Uvicorn may run handlers in a threadpool; the store is shared.
        self._lock = threading.Lock()

    def create(self) -> tuple[str, Conversation]:
        session_id = secrets.token_urlsafe(16)
        conversation = Conversation()
        with self._lock:
            self._entries[session_id] = _Entry(conversation, time.monotonic())
            self._evict()
        return session_id, conversation

    def get(self, session_id: str) -> Conversation | None:
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            if time.monotonic() - entry.last_used > self._ttl:
                del self._entries[session_id]
                return None
            entry.last_used = time.monotonic()
            self._entries.move_to_end(session_id)
            return entry.conversation

    def get_or_create(self, session_id: str | None
                      ) -> tuple[str, Conversation]:
        """Unknown or expired ids get a fresh session rather than a 404.

        A chat client that has been idle overnight should be able to
        keep typing; losing the thread is acceptable, an error is not.
        """
        if session_id:
            conversation = self.get(session_id)
            if conversation is not None:
                return session_id, conversation
        return self.create()

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._entries.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _evict(self) -> None:
        """Caller holds the lock."""
        now = time.monotonic()
        stale = [key for key, entry in self._entries.items()
                 if now - entry.last_used > self._ttl]
        for key in stale:
            del self._entries[key]
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)   # oldest first
