# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""LLM backend implementations for Cambrian."""

from cambrian.backends.base import LLMBackend
from cambrian.backends.openai_compat import OpenAICompatBackend

# AnthropicBackend is imported lazily to avoid requiring the anthropic package
# for users who don't need it. Import explicitly when needed:
#   from cambrian.backends.anthropic import AnthropicBackend

__all__ = ["LLMBackend", "OpenAICompatBackend"]
