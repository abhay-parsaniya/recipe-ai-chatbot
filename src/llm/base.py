"""Provider-agnostic LLM interface.

Nothing outside `src/llm/providers/` may import a vendor SDK. Everything
else in the project depends on this file only, so swapping Claude for
another provider is one new module plus one line in the registry.

The interface is deliberately tiny -- one generate() call, one
availability flag. Anything richer (tools, streaming, structured output)
would leak a particular provider's shape into the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Any provider failure, normalized.

    Callers must not have to know that Anthropic raises
    `anthropic.RateLimitError` while another vendor raises something
    else. Providers translate; callers catch this.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(ABC):
    """What the chatbot needs from any LLM."""

    name: str = "base"

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True if this provider can actually be called right now.

        Checked before use so a missing API key degrades to templates
        instead of raising on the first user message.
        """

    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResult:
        """One turn. Raises LLMError on any provider failure."""
