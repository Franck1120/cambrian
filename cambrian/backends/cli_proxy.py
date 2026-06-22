# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""CLIProxyBackend -- local proxy that exposes multiple frontier models.

The CLIProxy runs on localhost:8317 and forwards requests to:
- Antigravity (Gemini 3 Pro/Flash, GPT-OSS 120B)
- Codex (GPT-5, GPT-5.4)
- Gemini CLI (Gemini 2.5 Pro, 3 Pro, 3 Flash)

Usage::

    from cambrian.backends.cli_proxy import CLIProxyBackend

    backend = CLIProxyBackend(model="gemini-3-pro")
    reply = backend.generate("Hello!")
"""

from __future__ import annotations

from typing import Any

from cambrian.backends.openai_compat import OpenAICompatBackend

#: Default CLIProxy base URL.
CLI_PROXY_BASE_URL = "http://localhost:8317/v1"
#: Default CLIProxy API key.
CLI_PROXY_API_KEY = "jarvis-local-key"


class CLIProxyBackend(OpenAICompatBackend):
    """Backend for the local CLIProxy server (port 8317).

    Inherits all behaviour from
    :class:`~cambrian.backends.openai_compat.OpenAICompatBackend`.
    Defaults to ``base_url="http://localhost:8317/v1"`` and
    ``api_key="jarvis-local-key"``.

    Available models:

    - ``"gemini-3-pro"`` / ``"gemini-3-flash"`` / ``"gpt-oss-120b"`` -- Antigravity
    - ``"gpt-5"`` / ``"gpt-5.4"`` -- Codex
    - ``"gemini-2.5-pro"`` -- Gemini CLI

    Args:
        model: Model identifier. Default ``"gemini-3-pro"``.
        base_url: Override proxy URL. Default ``"http://localhost:8317/v1"``.
        api_key: Override API key. Default ``"jarvis-local-key"``.
        **kwargs: Forwarded to
            :class:`~cambrian.backends.openai_compat.OpenAICompatBackend`.
    """

    def __init__(
        self,
        model: str = "gemini-3-pro",
        base_url: str = CLI_PROXY_BASE_URL,
        api_key: str = CLI_PROXY_API_KEY,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
