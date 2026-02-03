"""LLM package: LiteLLM integration for provider-agnostic LLM access."""

from patina.llm.router import LLMRouter
from patina.llm.tools import MEMORY_TOOLS, get_memory_tools
from patina.llm.prompts import (
    DAILY_JOURNAL_PROMPT,
    WEEKLY_SYNTHESIS_PROMPT,
    MONTHLY_INTEGRATION_PROMPT,
)

__all__ = [
    "LLMRouter",
    "MEMORY_TOOLS",
    "get_memory_tools",
    "DAILY_JOURNAL_PROMPT",
    "WEEKLY_SYNTHESIS_PROMPT",
    "MONTHLY_INTEGRATION_PROMPT",
]
