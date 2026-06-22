# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""OpenAI-compatible HTTP backend.

Works with any API that speaks the OpenAI Chat Completions protocol:
- OpenAI (api.openai.com)
- Google Gemini (via OpenAI-compat endpoint)
- Ollama (localhost:11434)
- LM Studio (localhost:1234)
- vLLM
- Any other OpenAI-compatible server
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from cambrian.backends.base import LLMBackend

# -- Model catalogs -----------------------------------------------------------

#: Groq free-tier models (14 400 req/day). Use base_url="https://api.groq.com/openai/v1".
GROQ_MODELS: tuple[str, ...] = (
    "gpt-oss-120b",
    "gpt-oss-20b",
    "kimi-k2-instruct",  # 262K context window
    "qwen3-32b",
    "llama-4-scout",
    "llama-3.3-70b",
    "llama-3.1-8b",
)

#: CLIProxyAPI models (local proxy on port 8317).
#: Use base_url="http://localhost:8317/v1" and api_key="jarvis-local-key".
CLI_PROXY_MODELS: tuple[str, ...] = (
    # Antigravity provider
    "gemini-3-pro",
    "gemini-3-flash",
    "gpt-oss-120b",
    # Codex provider
    "gpt-5",
    "gpt-5.4",
    # Gemini CLI provider
    "gemini-2.5-pro",
)


class OpenAICompatBackend(LLMBackend):
    """HTTP backend for any OpenAI-compatible API.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``, ``"gemma3:8b"``).
        base_url: API base URL. Defaults to ``CAMBRIAN_BASE_URL`` env var,
            then ``https://api.openai.com/v1``.
        api_key: Bearer token. Defaults to ``CAMBRIAN_API_KEY`` or
            ``OPENAI_API_KEY`` env vars.
        temperature: Sampling temperature (0.0–2.0). Default ``0.7``.
        max_tokens: Maximum tokens to generate. Default ``2048``.
        timeout: HTTP request timeout in seconds. Default ``60``.
        max_retries: Number of retries on transient errors. Default ``3``.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._base_url = (
            base_url or os.getenv("CAMBRIAN_BASE_URL") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = (
            api_key
            or os.getenv("CAMBRIAN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        """Canonical model identifier as passed to the API (e.g. ``"gpt-4o-mini"``)."""
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Send *prompt* as a user message and return the assistant reply.

        Retries on HTTP 429 / 5xx with exponential back-off.
        """
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        system = kwargs.get("system", None)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        content=json.dumps(payload),
                    )
                if response.status_code == 200:
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"]).strip()

                # Retryable errors
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = 2**attempt
                    time.sleep(wait)
                    last_error = RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                    continue

                # Non-retryable
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )

            except httpx.TimeoutException as exc:
                last_error = exc
                time.sleep(2**attempt)
            except httpx.RequestError as exc:
                last_error = exc
                time.sleep(2**attempt)

        raise RuntimeError(
            f"Backend call failed after {self._max_retries} attempts: {last_error}"
        )


def groq_backend(
    model: str = "llama-3.3-70b",
    api_key: str | None = None,
    **kwargs: Any,
) -> OpenAICompatBackend:
    """Create an :class:`OpenAICompatBackend` pre-configured for Groq.

    Args:
        model: One of :data:`GROQ_MODELS`. Default ``"llama-3.3-70b"``.
        api_key: Groq API key. Falls back to ``GROQ_API_KEY`` env var.
        **kwargs: Forwarded to :class:`OpenAICompatBackend`.

    Returns:
        Configured backend ready to call the Groq API.
    """
    resolved_key = api_key or os.getenv("GROQ_API_KEY") or ""
    return OpenAICompatBackend(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key=resolved_key,
        **kwargs,
    )
