"""Optional LLM response-generation layer.

Retrieval lives in `src.search` and runs with or without this package.
"""

from src.llm.base import LLMError, LLMProvider, LLMResult
from src.llm.service import GeneratedResponse, ResponseGenerator, build_provider

__all__ = [
    "LLMProvider",
    "LLMResult",
    "LLMError",
    "ResponseGenerator",
    "GeneratedResponse",
    "build_provider",
]
