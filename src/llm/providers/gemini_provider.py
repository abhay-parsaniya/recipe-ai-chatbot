"""Google Gemini implementation of LLMProvider.

Chosen for the free tier. Everything Google-specific is confined to this
file: `src/llm/base.py` and every caller above it know only about
`LLMProvider`, `LLMResult` and `LLMError`, so switching provider is one
module plus one line in `build_provider`.

Free-tier notes that shape the code below:

* Rate limits are per-minute and low. A 429 is an ordinary event here,
  not an emergency, so it is translated to a retryable LLMError and the
  chat degrades to template wording for that turn rather than erroring.
* The free tier has no SLA. Timeouts are kept short for the same reason
  -- a recipe app should not hang for 30 seconds waiting on a free API.
"""

from __future__ import annotations

from src.config.settings import Settings, get_settings
from src.llm.base import LLMError, LLMProvider, LLMResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None          # built lazily; needs a key
        # Models differ on whether they accept thinking_config. Learning
        # that costs one rejected request; repeating it every turn would
        # double our usage against a per-day free-tier quota.
        self._thinking_supported = True

    # -- availability ---------------------------------------------------

    @property
    def api_key(self) -> str | None:
        """From settings, which reads GOOGLE_API_KEY (or GEMINI_API_KEY)
        out of the environment. Never a literal in this repo."""
        secret = self.settings.google_api_key
        return secret.get_secret_value() if secret else None

    @property
    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import google.genai  # noqa: F401
        except ImportError:
            logger.warning("google-genai not installed; Gemini disabled")
            return False
        return True

    @property
    def client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.settings.llm_timeout_seconds * 1000),
                ),
            )
        return self._client

    # -- generation -----------------------------------------------------

    def generate(self, system: str, user: str) -> LLMResult:
        from google.genai import errors, types

        if not self.api_key:
            raise LLMError("GOOGLE_API_KEY is not set")

        try:
            response = self._call(system, user,
                                  thinking=self._thinking_supported)
        except errors.ClientError as exc:
            # Not every model accepts thinking_config -- flash-lite
            # returns 400 INVALID_ARGUMENT for it. Retry once without,
            # so a model swap does not need a code change.
            if self._thinking_supported and self._rejects_thinking(exc):
                logger.info("%s rejects thinking_config; retrying without "
                            "and remembering", self.settings.llm_model)
                self._thinking_supported = False
                try:
                    response = self._call(system, user, thinking=False)
                except errors.APIError as retry_exc:
                    raise LLMError(f"Gemini API error: {retry_exc}") from retry_exc
            else:
                raise self._client_error(exc) from exc
        except errors.ServerError as exc:
            # 4xx. 429 is a free-tier rate limit and is worth retrying;
            # 400/401/403 are configuration faults and are not.
            raise LLMError(f"Gemini server error: {exc}", retryable=True) from exc
        except errors.APIError as exc:
            raise LLMError(f"Gemini API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network stack, DNS, TLS
            raise LLMError(f"Gemini call failed: {exc}", retryable=True) from exc

        text = self._extract_text(response)
        if not text:
            raise LLMError(f"Gemini returned no text "
                           f"({self._finish_reason(response)})")

        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            text=text,
            model=self.settings.llm_model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )

    def _call(self, system: str, user: str, thinking: bool):
        from google.genai import types

        config = dict(
            # The grounding contract goes here, not in `contents`, so the
            # user's message can never be mistaken for instructions.
            system_instruction=system,
            max_output_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )
        budget = self.settings.llm_thinking_budget
        if thinking and budget >= 0:
            config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget)

        return self.client.models.generate_content(
            model=self.settings.llm_model,
            contents=user,
            config=types.GenerateContentConfig(**config),
        )

    @staticmethod
    def _rejects_thinking(exc) -> bool:
        """Any 400 while thinking_config was attached is worth one retry
        without it.

        Do NOT try to recognise the message: Gemini returns a bare
        "Request contains an invalid argument." with no mention of the
        offending field. An earlier version matched on the word
        "thinking", never fired, and silently disabled the LLM for every
        model that rejects the parameter.

        A 400 caused by something else costs one wasted retry and then
        surfaces normally, which is the cheaper failure of the two.
        """
        code = getattr(exc, "code", None) or getattr(exc, "status", None)
        return code == 400

    @staticmethod
    def _client_error(exc) -> LLMError:
        """4xx. 429 is a free-tier rate limit and worth retrying; 400,
        401 and 403 are configuration faults and are not."""
        code = getattr(exc, "code", None) or getattr(exc, "status", None)
        return LLMError(f"Gemini rejected the request ({code}): {exc}",
                        retryable=code == 429)

    # -- response handling ----------------------------------------------

    @staticmethod
    def _extract_text(response) -> str:
        """`response.text` is a convenience that raises or warns on a
        blocked or empty candidate, so the parts are walked directly."""
        text = ""
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text += getattr(part, "text", "") or ""
        return text.strip()

    @staticmethod
    def _finish_reason(response) -> str:
        """Why nothing came back -- usually a safety block, which the
        caller logs before falling back to the template."""
        for candidate in getattr(response, "candidates", None) or []:
            reason = getattr(candidate, "finish_reason", None)
            if reason:
                return f"finish_reason={reason}"
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            return f"prompt_feedback={feedback}"
        return "no candidates returned"
