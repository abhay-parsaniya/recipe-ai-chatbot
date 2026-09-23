"""The generation layer.

Sits *after* retrieval and knows nothing about how retrieval works. It
receives recipes that were already found, and turns them into prose.

Three guarantees it provides to the caller:

1. **It never runs retrieval.** Recipes come in as an argument.
2. **It never raises.** Every failure -- no key, no network, rate limit,
   bad model name, empty reply -- returns the template text instead.
   A recipe app that 500s because a third party is down is a bad app.
3. **It checks the output before returning it.** Prompting is not proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config.settings import Settings, get_settings
from src.llm.base import LLMError, LLMProvider
from src.llm.prompts import (
    SYSTEM_PROMPT,
    build_followup_message,
    build_user_message,
)
from src.search import SearchResult
from src.utils.logging import get_logger
from src.utils.text import normalize

logger = get_logger(__name__)


def build_provider(settings: Settings | None = None) -> LLMProvider | None:
    """Registry. Adding a provider means adding one line here.

    Returns None when the LLM is switched off or the provider name is
    unknown -- both are normal states, not errors.
    """
    settings = settings or get_settings()
    if not settings.llm_enabled:
        return None

    if settings.llm_provider == "anthropic":
        from src.llm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)

    logger.warning("Unknown llm_provider %r; running without an LLM",
                   settings.llm_provider)
    return None


@dataclass
class GeneratedResponse:
    """Carries *how* the text was produced, so the UI can be honest."""

    text: str
    used_llm: bool
    fallback_reason: str = ""


class ResponseGenerator:
    """Phrase a reply from recipes that retrieval already selected."""

    def __init__(self, provider: LLMProvider | None = None,
                 settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = provider if provider is not None else build_provider(
            self.settings
        )

    @property
    def enabled(self) -> bool:
        return self.provider is not None and self.provider.is_available

    # -- public API -----------------------------------------------------

    def generate_search_reply(self, user_query: str,
                              results: list[SearchResult],
                              fallback_text: str,
                              history: str = "") -> GeneratedResponse:
        """Phrase a set of search results."""
        return self._generate(
            user_message=build_user_message(user_query, results, history),
            fallback_text=fallback_text,
            grounding=results,
        )

    def generate_followup_reply(self, user_query: str,
                                selected: SearchResult,
                                fallback_text: str,
                                history: str = "") -> GeneratedResponse:
        """Phrase an answer about one selected recipe."""
        return self._generate(
            user_message=build_followup_message(user_query, selected, history),
            fallback_text=fallback_text,
            grounding=[selected],
        )

    # -- internals ------------------------------------------------------

    def _generate(self, user_message: str, fallback_text: str,
                  grounding: list[SearchResult]) -> GeneratedResponse:
        if not self.enabled:
            return GeneratedResponse(fallback_text, used_llm=False,
                                     fallback_reason="llm disabled")

        try:
            result = self.provider.generate(SYSTEM_PROMPT, user_message)
        except LLMError as exc:
            # Requirement: degrade, never crash. The user still gets
            # their recipes; only the phrasing is plainer.
            logger.warning("LLM call failed (%s); using template", exc)
            return GeneratedResponse(fallback_text, used_llm=False,
                                     fallback_reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            # A provider bug must not take the chat down either.
            logger.exception("Unexpected LLM failure; using template")
            return GeneratedResponse(fallback_text, used_llm=False,
                                     fallback_reason=f"unexpected: {exc}")

        problem = self._ungrounded_reason(result.text, grounding)
        if problem:
            logger.warning("Rejecting LLM output: %s", problem)
            return GeneratedResponse(fallback_text, used_llm=False,
                                     fallback_reason=problem)

        logger.info("LLM reply: %d in / %d out tokens",
                    result.input_tokens, result.output_tokens)
        return GeneratedResponse(result.text, used_llm=True)

    @staticmethod
    def _ungrounded_reason(text: str, grounding: list[SearchResult]) -> str:
        """A cheap check that the reply stayed inside the context.

        This is not verification -- proving a paragraph is faithful to a
        source needs another model. It catches the one failure that
        matters most here and is mechanically detectable: the reply
        presenting a recipe *title* that was never retrieved. Titles are
        what a user will act on, and a hallucinated one is invisible in
        prose.

        Deliberately narrow, so it does not fire on ordinary paraphrase.
        """
        if not text.strip():
            return "empty reply"
        if not grounding:
            # With no recipes retrieved, the model should be admitting
            # that. A confident ingredient list here means invention.
            if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
                return "numbered recipe steps with no retrieved recipes"
            return ""

        # Quoted strings are the model's own claim of a title.
        quoted = re.findall(r'"([^"]{4,80})"', text)
        known = {normalize(r.title) for r in grounding}
        for candidate in quoted:
            normalized = normalize(candidate)
            if not normalized:
                continue
            if any(normalized in title or title in normalized
                   for title in known):
                continue
            # Quoting the user's own words back is fine; only flag
            # things that look like a dish name we never supplied.
            if len(normalized.split()) >= 2:
                return f"cited an unretrieved title: {candidate!r}"
        return ""
