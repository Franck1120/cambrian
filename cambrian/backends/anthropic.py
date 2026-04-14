"""Anthropic Claude backend for Cambrian.

Uses the official ``anthropic`` Python SDK to call the Claude API.
Supports all Claude models (claude-opus-4-6, claude-sonnet-4-6,
claude-haiku-4-5, etc.) and honours Cambrian's standard ``generate``
kwargs (``system``, ``temperature``, ``max_tokens``).

Setup::

    pip install anthropic

    export ANTHROPIC_API_KEY=sk-ant-...
    # or pass api_key= directly

Usage::

    from cambrian.backends.anthropic import AnthropicBackend

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001")
    response = backend.generate("Write a haiku about evolution.")
"""

from __future__ import annotations

import os
import time
from typing import Any

from cambrian.backends.base import LLMBackend


class AnthropicBackend(LLMBackend):
    """LLM backend that calls the Anthropic Messages API via the official SDK.

    Args:
        model: Anthropic model identifier.
            Default ``"claude-haiku-4-5-20251001"`` (fast + cheap).
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        temperature: Default sampling temperature (0.0–1.0 for Claude).
            Default ``0.7``.
        max_tokens: Maximum tokens to generate. Default ``2048``.
        max_retries: Retries on overload / server errors. Default ``3``.
        timeout: Per-request timeout in seconds. Default ``60``.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        try:
            import anthropic as _anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicBackend. "
                "Install it with: pip install anthropic"
            ) from exc

        self._model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or ""
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        """Canonical model identifier (e.g. ``"claude-haiku-4-5-20251001"``)."""
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send *prompt* to the Anthropic Messages API and return the reply.

        Args:
            prompt: User-turn message content.
            **kwargs: Overrides — ``system`` (str), ``temperature`` (float),
                ``max_tokens`` (int).

        Returns:
            Assistant reply as a plain string.

        Raises:
            RuntimeError: If the API call fails after *max_retries* attempts.
            ImportError: If the ``anthropic`` package is not installed.
        """
        import anthropic

        temperature = float(kwargs.get("temperature", self._temperature))
        max_tokens = int(kwargs.get("max_tokens", self._max_tokens))
        system = kwargs.get("system", None)

        # Claude temperature is 0.0–1.0; clamp silently
        temperature = max(0.0, min(1.0, temperature))

        client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=self._timeout,
        )

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            create_kwargs["system"] = system

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                message = client.messages.create(**create_kwargs)
                # Extract text from the first content block
                for block in message.content:
                    if hasattr(block, "text"):
                        return block.text.strip()
                return ""

            except anthropic.RateLimitError as exc:
                last_error = exc
                time.sleep(2 ** attempt)
            except anthropic.APIStatusError as exc:
                if exc.status_code in (500, 502, 503, 529):
                    last_error = exc
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(
                        f"Anthropic API error {exc.status_code}: {exc.message}"
                    ) from exc
            except anthropic.APIConnectionError as exc:
                last_error = exc
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"AnthropicBackend failed after {self._max_retries} attempts: {last_error}"
        )

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self._model!r})"
