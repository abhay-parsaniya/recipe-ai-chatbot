"""Claude (Anthropic) implementation of LLMProvider."""

from __future__ import annotations

from src.config.settings import Settings, get_settings
from src.llm.base import LLMError, LLMProvider, LLMResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None  # built lazily; constructing it needs a key

    # -- availability ---------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """Read from settings, which reads ANTHROPIC_API_KEY from the
        environment. The key is never a literal anywhere in this repo."""
        secret = self.settings.anthropic_api_key
        return secret.get_secret_value() if secret else None

    @property
    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.warning("anthropic package not installed; LLM disabled")
            return False
        return True

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.settings.llm_timeout_seconds,
                # The SDK already retries 429/5xx with backoff. Leaving
                # this at the default rather than hand-rolling a loop.
                max_retries=2,
            )
        return self._client

    # -- generation -----------------------------------------------------

    def generate(self, system: str, user: str) -> LLMResult:
        import anthropic

        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

        try:
            response = self.client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Rephrasing supplied text is not a reasoning problem;
                # low effort keeps latency and cost down. Thinking is
                # left on (the default on Opus 5) -- disabling it can
                # make the model leak stray tags into the reply.
                output_config={"effort": self.settings.llm_effort},
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError(f"Invalid API key: {exc}") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"Key lacks permission: {exc}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Unknown model {self.settings.llm_model}: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Bad request: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Rate limited: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc}",
                           retryable=exc.status_code >= 500) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Network error: {exc}", retryable=True) from exc

        if response.stop_reason == "refusal":
            raise LLMError("Model declined to answer")

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise LLMError("Model returned no text")

        return LLMResult(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
