"""Test isolation.

Without this, the suite reads whatever `.env` the developer happens to
have. The moment a real GOOGLE_API_KEY and LLM_ENABLED=true landed in
`.env`, two tests that assert "the LLM is off by default" began failing
-- not because the code changed, but because the machine did. Worse, the
opposite case is silent: a test that should exercise the template path
would quietly start calling a paid API.

So every Settings() built during tests ignores `.env` and the ambient
environment, and sees only explicit arguments plus the declared
defaults.
"""

from __future__ import annotations

import pytest

from src.config.settings import Settings, get_settings

# Variables that would otherwise leak in from the developer's shell.
LEAKY = (
    "LLM_ENABLED", "LLM_PROVIDER", "LLM_MODEL",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
    "MAX_RECIPES", "COVERAGE_WEIGHT", "DEFAULT_TOP_K",
)


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Make Settings() deterministic for the whole suite."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setitem(Settings.model_config, "secrets_dir", None)
    for name in LEAKY:
        monkeypatch.delenv(name, raising=False)

    # get_settings() is lru_cached; a value cached before this fixture
    # ran would carry the developer's .env into every later test.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
