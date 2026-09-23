"""The chatbot: route a message to an action, produce a reply.

Deliberately thin. Understanding lives in `intents` and `query`, wording
lives in `responses`, memory lives in `history`, retrieval lives in
`search`. This file only decides *what to do*, which is why it fits on
one screen and why swapping any one layer for an LLM is a local change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.chatbot import responses
from src.chatbot.guardrails import classify
from src.chatbot.history import Conversation
from src.chatbot.intents import Intent, ParsedMessage, detect_intent, extract_position
from src.chatbot.pantry_query import looks_like_pantry_request, parse_pantry
from src.chatbot.query import build_query, extract_terms, has_negation
from src.config.settings import Settings, get_settings
from src.llm import ResponseGenerator
from src.search import RecipeSearch, SearchResult
from src.search.pantry import PantryMatcher
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BotReply:
    """A reply plus the state behind it, so callers can render properly.

    The API and the Streamlit UI both need the structured results, not
    only the text -- a UI should show cards, not a preformatted list.
    """

    text: str
    intent: Intent
    results: list[SearchResult] = field(default_factory=list)
    selected: SearchResult | None = None
    # True when an LLM phrased this reply. Surfaced so a UI can label it
    # and so tests can assert which path ran.
    used_llm: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "intent": self.intent.value,
            "results": [r.to_dict() for r in self.results],
            "selected": self.selected.to_dict() if self.selected else None,
            "used_llm": self.used_llm,
        }


class RecipeChatbot:
    """Stateless with respect to the engine, stateful per Conversation.

    One engine (big, shared, read-only) serves many conversations
    (small, per-user). That split is what lets the API hold a single
    index in memory and still give every caller their own thread.
    """

    def __init__(self, engine: RecipeSearch, settings: Settings | None = None,
                 generator: ResponseGenerator | None = None):
        self.engine = engine
        self.settings = settings or get_settings()
        # The generator is a phrasing layer only. If it is absent or
        # disabled the bot behaves exactly as before -- retrieval and
        # routing do not consult it.
        self.generator = generator or ResponseGenerator(settings=self.settings)
        # Shares the fitted engine; holds no state of its own.
        self.pantry = PantryMatcher(engine, self.settings)

    @classmethod
    def from_index(cls, settings: Settings | None = None) -> "RecipeChatbot":
        settings = settings or get_settings()
        return cls(RecipeSearch.load(settings=settings), settings)

    @property
    def llm_enabled(self) -> bool:
        return self.generator.enabled

    def _history_for_llm(self, conversation: Conversation, turns: int = 4) -> str:
        """A short transcript tail, for pronoun resolution only.

        Trimmed hard: long histories cost tokens and give the model more
        of its own earlier prose to echo, which is exactly where
        invented detail creeps back in.
        """
        recent = conversation.transcript(limit=turns * 2)[:-1]
        return "\n".join(f"{m.role}: {m.text[:300]}" for m in recent)

    # -- understanding --------------------------------------------------

    def parse(self, message: str, conversation: Conversation) -> ParsedMessage:
        intent = detect_intent(
            message,
            has_results=conversation.has_results,
            has_selection=conversation.has_selection,
        )

        # The domain check gates SEARCH only. Conversation control --
        # "the second one", "how do I make it", "bye" -- is about
        # recipes already on screen and contains no food words for the
        # guardrail to find, so running it there would reject valid
        # follow-ups. Anything that would otherwise become a search is
        # checked, whether or not results are on screen.
        if intent in (Intent.SEARCH, Intent.UNKNOWN):
            verdict = classify(message)
            if not verdict:
                return ParsedMessage(intent=Intent.OUT_OF_DOMAIN, text=message,
                                     terms=[verdict.category.value])

            # "I have X but no Y" is an inventory, not a query. It needs
            # the opposite scoring -- how much of the recipe you have,
            # not how much of your words the recipe has -- so it routes
            # to a different handler rather than a flag on search.
            if looks_like_pantry_request(message):
                have, exclude = parse_pantry(message)
                if have:
                    return ParsedMessage(intent=Intent.PANTRY, text=message,
                                         have=have, exclude=exclude)
        return ParsedMessage(
            intent=intent,
            text=message,
            position=extract_position(message) if intent is Intent.SELECT else None,
            terms=extract_terms(message) if intent is Intent.SEARCH else [],
        )

    # -- main entry point -----------------------------------------------

    def respond(self, message: str, conversation: Conversation) -> BotReply:
        """Handle one user turn and record both sides in the transcript."""
        conversation.add_user(message)
        parsed = self.parse(message, conversation)
        logger.info("intent=%s message=%r", parsed.intent.value, message)

        handler = self._HANDLERS[parsed.intent]
        reply = handler(self, parsed, conversation)

        conversation.add_bot(reply.text)
        return reply

    # -- handlers -------------------------------------------------------

    def _handle_search(self, parsed: ParsedMessage,
                       conversation: Conversation) -> BotReply:
        query = build_query(parsed.text)
        top_k = self.settings.default_top_k
        results = self.engine.search(query, top_k=top_k)
        conversation.set_results(results, query)

        text = responses.format_results(results, query)
        if results and has_negation(parsed.text):
            text += "\n" + responses.NEGATION_CAVEAT

        # Generation happens strictly after retrieval, and only ever
        # sees the recipes retrieval already chose.
        generated = self.generator.generate_search_reply(
            user_query=parsed.text,
            results=results,
            fallback_text=text,
            history=self._history_for_llm(conversation),
        )
        return BotReply(text=generated.text, intent=Intent.SEARCH,
                        results=results, used_llm=generated.used_llm)

    def _handle_pantry(self, parsed: ParsedMessage,
                       conversation: Conversation) -> BotReply:
        """Rank by recipe completeness instead of query similarity."""
        if not parsed.have:
            return BotReply(text=responses.NO_PANTRY_ITEMS,
                            intent=Intent.PANTRY)

        matches = self.pantry.match(
            have=parsed.have,
            exclude=parsed.exclude,
            top_k=self.settings.default_top_k,
        )

        # Selection and follow-ups work on pantry hits too, so the
        # matches are converted back into ordinary results and stored.
        results = [self._to_result(m) for m in matches]
        conversation.set_results(results, ", ".join(parsed.have))

        template = responses.format_pantry(matches, parsed.have,
                                           parsed.exclude)
        generated = self.generator.generate_search_reply(
            user_query=parsed.text,
            results=results,
            fallback_text=template,
            history=self._history_for_llm(conversation),
        )
        return BotReply(text=generated.text, intent=Intent.PANTRY,
                        results=results, used_llm=generated.used_llm)

    @staticmethod
    def _to_result(match) -> SearchResult:
        """A PantryMatch as a SearchResult, so the rest of the bot,
        the API and the UI need no special case for pantry hits."""
        return SearchResult(
            recipe_id=match.recipe_id,
            title=match.title,
            score=match.match,
            similarity=match.match,
            coverage=match.match,
            ingredients=match.ingredients,
            directions=match.directions,
            ner=match.have,
            link=match.link,
            matched_terms=match.have,
        )

    def _handle_more(self, parsed: ParsedMessage,
                     conversation: Conversation) -> BotReply:
        if not conversation.has_results:
            return BotReply(text=responses.NO_SEARCH_YET, intent=Intent.MORE)
        return self._next_page(conversation)

    def _next_page(self, conversation: Conversation) -> BotReply:
        """Fetch the next page.

        The engine has no paging, so we ask for offset+k and slice. Fine
        at these depths; a real pager would push the offset down into
        _top_indices.
        """
        top_k = self.settings.default_top_k
        wanted = conversation.offset + len(conversation.last_results) + top_k
        page = self.engine.search(conversation.last_query, top_k=wanted)
        already = conversation.offset + len(conversation.last_results)
        results = page[already:]

        if not results:
            return BotReply(text="That's everything I have for that search.",
                            intent=Intent.MORE)

        conversation.extend_results(results)
        return BotReply(
            text=responses.format_results(results, conversation.last_query,
                                          offset=conversation.offset),
            intent=Intent.MORE,
            results=results,
        )

    def _handle_select(self, parsed: ParsedMessage,
                       conversation: Conversation) -> BotReply:
        if not conversation.has_results:
            return BotReply(text=responses.NO_SEARCH_YET, intent=Intent.SELECT)
        if parsed.position is None:
            return BotReply(text=responses.NOTHING_SELECTED, intent=Intent.SELECT)
        selected = conversation.select(parsed.position)
        if selected is None:
            count = len(conversation.last_results)
            return BotReply(
                text=f"I only showed {count} recipes -- pick 1 to {count}.",
                intent=Intent.SELECT,
            )
        generated = self.generator.generate_followup_reply(
            user_query=parsed.text,
            selected=selected,
            fallback_text=responses.format_selection(selected),
            history=self._history_for_llm(conversation),
        )
        return BotReply(text=generated.text, intent=Intent.SELECT,
                        selected=selected, used_llm=generated.used_llm)

    def _detail(self, conversation: Conversation, intent: Intent,
                formatter, message: str = "") -> BotReply:
        """Shared path for ingredients / directions / link.

        If exactly one recipe is on screen, treat it as selected: making
        the user say "1" before asking "how do I make it" is pedantry.
        """
        selected = conversation.selected
        if selected is None and len(conversation.last_results) == 1:
            selected = conversation.select(1)
        if selected is None:
            text = (responses.NOTHING_SELECTED if conversation.has_results
                    else responses.NO_SEARCH_YET)
            return BotReply(text=text, intent=intent)

        template = formatter(selected)
        generated = self.generator.generate_followup_reply(
            user_query=message,
            selected=selected,
            fallback_text=template,
            history=self._history_for_llm(conversation),
        )
        return BotReply(text=generated.text, intent=intent,
                        selected=selected, used_llm=generated.used_llm)

    def _handle_ingredients(self, parsed, conversation) -> BotReply:
        return self._detail(conversation, Intent.INGREDIENTS,
                            responses.format_ingredients, parsed.text)

    def _handle_directions(self, parsed, conversation) -> BotReply:
        return self._detail(conversation, Intent.DIRECTIONS,
                            responses.format_directions, parsed.text)

    def _handle_link(self, parsed, conversation) -> BotReply:
        return self._detail(conversation, Intent.LINK, responses.format_link)

    def _handle_reset(self, parsed, conversation) -> BotReply:
        conversation.reset()
        return BotReply(text="Cleared. What ingredients do you have?",
                        intent=Intent.RESET)

    def _handle_greet(self, parsed, conversation) -> BotReply:
        return BotReply(text=responses.GREETING, intent=Intent.GREET)

    def _handle_help(self, parsed, conversation) -> BotReply:
        return BotReply(text=responses.HELP, intent=Intent.HELP)

    def _handle_goodbye(self, parsed, conversation) -> BotReply:
        return BotReply(text=responses.GOODBYE, intent=Intent.GOODBYE)

    def _handle_unknown(self, parsed, conversation) -> BotReply:
        return BotReply(text=responses.UNKNOWN, intent=Intent.UNKNOWN)

    def _handle_out_of_domain(self, parsed, conversation) -> BotReply:
        """Refuse without answering, and without calling the LLM.

        No retrieval happens, so there is nothing for a model to
        ground against -- the only safe reply is a fixed one.
        """
        category = parsed.terms[0] if parsed.terms else "other"
        return BotReply(text=responses.format_out_of_domain(category),
                        intent=Intent.OUT_OF_DOMAIN)

    _HANDLERS = {
        Intent.SEARCH: _handle_search,
        Intent.PANTRY: _handle_pantry,
        Intent.MORE: _handle_more,
        Intent.SELECT: _handle_select,
        Intent.INGREDIENTS: _handle_ingredients,
        Intent.DIRECTIONS: _handle_directions,
        Intent.LINK: _handle_link,
        Intent.RESET: _handle_reset,
        Intent.GREET: _handle_greet,
        Intent.HELP: _handle_help,
        Intent.GOODBYE: _handle_goodbye,
        Intent.OUT_OF_DOMAIN: _handle_out_of_domain,
        Intent.UNKNOWN: _handle_unknown,
    }
