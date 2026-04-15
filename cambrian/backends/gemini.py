# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Google Gemini backend for Cambrian.

Uses the official ``google-genai`` Python SDK (``google.genai``) to call the
Gemini API.  Supports all Gemini models (gemini-2.0-flash, gemini-1.5-pro,
gemini-1.5-flash, etc.) and honours Cambrian's standard ``generate``
kwargs (``system``, ``temperature``, ``max_tokens``).

Setup::

    pip install google-genai

    export GEMINI_API_KEY=AIza...
    # or pass api_key= directly

Usage::

    from cambrian.backends.gemini import GeminiBackend

    backend = GeminiBackend(model="gemini-2.0-flash")
    response = backend.generate("Write a haiku about evolution.")

    # With system prompt:
    response = backend.generate(
        "Sort this list: [3, 1, 2]",
        system="You are a Python expert. Always return only valid Python.",
    )
"""

from __future__ import annotations

import os
import time
from typing import Any

from cambrian.backends.base import LLMBackend


class GeminiBackend(LLMBackend):
    """LLM backend that calls the Google Gemini API via the official SDK.

    Args:
        model: Gemini model identifier.
            Default ``"gemini-2.0-flash"`` (fast, cost-effective).
        api_key: Google AI Studio API key.  Falls back to ``GEMINI_API_KEY``
            then ``GOOGLE_API_KEY`` environment variables.
        temperature: Default sampling temperature (0.0–2.0 for Gemini).
            Default ``0.7``.
        max_tokens: Maximum tokens to generate (``max_output_tokens``).
            Default ``2048``.
        max_retries: Retries on rate-limit / server errors. Default ``3``.
        timeout: Per-request timeout in seconds. Default ``60``.

    Raises:
        ImportError: If the ``google-genai`` package is not installed.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

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
            import google.genai as _genai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for GeminiBackend. "
                "Install it with: pip install google-genai"
            ) from exc

        self._model = model
        self._api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        """Canonical model identifier (e.g. ``"gemini-2.0-flash"``)."""
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send *prompt* to the Gemini API and return the reply.

        Args:
            prompt: User-turn message content.
            **kwargs: Overrides — ``system`` (str), ``temperature`` (float),
                ``max_tokens`` (int).

        Returns:
            Model reply as a plain string.

        Raises:
            RuntimeError: If the API call fails after *max_retries* attempts.
            ImportError: If the ``google-genai`` package is not installed.
        """
        import google.genai as genai
        from google.genai import types as genai_types

        temperature = float(kwargs.get("temperature", self._temperature))
        max_tokens = int(kwargs.get("max_tokens", self._max_tokens))
        system_instruction = kwargs.get("system", None)

        # Gemini temperature range: 0.0–2.0
        temperature = max(0.0, min(2.0, temperature))

        client = genai.Client(api_key=self._api_key)

        generate_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=generate_config,
                )
                # Extract text from the response
                if response.text is not None:
                    return str(response.text).strip()
                # Fallback: concatenate all text parts
                parts: list[str] = []
                for candidate in response.candidates or []:
                    content = candidate.content
                    if content is None:
                        continue
                    for part in (content.parts or []):
                        if hasattr(part, "text") and part.text:
                            parts.append(str(part.text))
                return "".join(parts).strip()

            except Exception as exc:
                exc_name = type(exc).__name__
                # Retry on quota / server errors
                if any(
                    kw in exc_name.lower()
                    for kw in ("ratelimit", "quota", "resource", "unavailable", "internal")
                ):
                    last_error = exc
                    time.sleep(2 ** attempt)
                    continue
                # Check message text for retryable signals
                exc_str = str(exc).lower()
                if any(
                    kw in exc_str
                    for kw in ("429", "quota", "rate limit", "503", "500", "unavailable")
                ):
                    last_error = exc
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini API error: {exc}") from exc

        raise RuntimeError(
            f"GeminiBackend failed after {self._max_retries} attempts: {last_error}"
        )

    def __repr__(self) -> str:
        return f"GeminiBackend(model={self._model!r})"
