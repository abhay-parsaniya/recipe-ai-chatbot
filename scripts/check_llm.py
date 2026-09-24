"""Verify the LLM layer against the real API, and show the difference.

Run:  python -m scripts.check_llm ["your question"]

Prints the template reply and the LLM reply side by side so you can see
what the model actually changed, plus token usage and latency. Exits
non-zero if the LLM path did not work, so this is usable as a smoke test.
"""

from __future__ import annotations

import sys
import time

from src.chatbot import Conversation, RecipeChatbot
from src.config.settings import get_settings
from src.llm import ResponseGenerator, build_provider

RULE = "=" * 72


def main() -> int:
    question = " ".join(sys.argv[1:]) or \
        "what can I make with chicken, tomato and onion?"
    settings = get_settings()

    print(f"provider : {settings.llm_provider}")
    print(f"model    : {settings.llm_model}")
    print(f"enabled  : {settings.llm_enabled}")

    provider = build_provider(settings)
    if provider is None:
        print("\nLLM is off. Set LLM_ENABLED=true in .env")
        return 1
    if not provider.is_available:
        key = "GOOGLE_API_KEY" if settings.llm_provider == "gemini" \
            else "ANTHROPIC_API_KEY"
        print(f"\nProvider unavailable — is {key} set in .env?")
        return 1

    # 1. a direct call, so a broken key fails loudly here rather than
    #    being silently swallowed by the template fallback
    print(f"\n{RULE}\nDIRECT CALL\n{RULE}")
    started = time.perf_counter()
    try:
        result = provider.generate(
            "You are a terse assistant. Reply with exactly: OK",
            "Say OK.")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"reply    : {result.text[:80]!r}")
    print(f"tokens   : {result.input_tokens} in / {result.output_tokens} out")
    print(f"latency  : {(time.perf_counter() - started) * 1000:.0f} ms")

    # 2. the real thing, through the bot
    bot = RecipeChatbot.from_index(settings)
    print(f"\n{RULE}\nTEMPLATE (LLM off)\n{RULE}")
    plain = RecipeChatbot(bot.engine, settings,
                          ResponseGenerator(provider=None, settings=settings))
    print(plain.respond(question, Conversation()).text)

    print(f"\n{RULE}\nLLM ({settings.llm_model})\n{RULE}")
    started = time.perf_counter()
    reply = bot.respond(question, Conversation())
    elapsed = (time.perf_counter() - started) * 1000
    print(reply.text)
    print(f"\nused_llm : {reply.used_llm}   ({elapsed:.0f} ms, "
          f"{len(reply.results)} recipes retrieved)")

    if not reply.used_llm:
        print("\nThe reply fell back to the template — check the log line "
              "'LLM call failed' or 'Rejecting LLM output' above.")
        return 1

    print("\nLLM path verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
