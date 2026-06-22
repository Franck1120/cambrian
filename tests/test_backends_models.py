# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Tests for Groq/CLIProxy model catalogs, groq_backend factory, and CLIProxyBackend."""

from __future__ import annotations

import pytest

from cambrian.backends.openai_compat import (
    CLI_PROXY_MODELS,
    GROQ_MODELS,
    OpenAICompatBackend,
    groq_backend,
)
from cambrian.backends.cli_proxy import (
    CLI_PROXY_API_KEY,
    CLI_PROXY_BASE_URL,
    CLIProxyBackend,
)
from cambrian.backends.base import LLMBackend
from cambrian.router import (
    GROQ_CHEAP_MODEL,
    GROQ_MEDIUM_MODEL,
    CLI_PROXY_PREMIUM_MODEL,
)


# ---------------------------------------------------------------------------
# GROQ_MODELS catalog
# ---------------------------------------------------------------------------


class TestGroqModelsCatalog:
    """Verify the GROQ_MODELS constant is correctly defined."""

    def test_is_tuple(self) -> None:
        assert isinstance(GROQ_MODELS, tuple)

    def test_not_empty(self) -> None:
        assert len(GROQ_MODELS) > 0

    def test_contains_expected_models(self) -> None:
        expected = {
            "gpt-oss-120b",
            "gpt-oss-20b",
            "kimi-k2-instruct",
            "qwen3-32b",
            "llama-4-scout",
            "llama-3.3-70b",
            "llama-3.1-8b",
        }
        assert set(GROQ_MODELS) == expected

    def test_all_entries_are_strings(self) -> None:
        for model in GROQ_MODELS:
            assert isinstance(model, str)

    def test_no_duplicates(self) -> None:
        assert len(GROQ_MODELS) == len(set(GROQ_MODELS))


# ---------------------------------------------------------------------------
# CLI_PROXY_MODELS catalog
# ---------------------------------------------------------------------------


class TestCLIProxyModelsCatalog:
    """Verify the CLI_PROXY_MODELS constant is correctly defined."""

    def test_is_tuple(self) -> None:
        assert isinstance(CLI_PROXY_MODELS, tuple)

    def test_not_empty(self) -> None:
        assert len(CLI_PROXY_MODELS) > 0

    def test_contains_expected_models(self) -> None:
        expected = {
            "gemini-3-pro",
            "gemini-3-flash",
            "gpt-oss-120b",
            "gpt-5",
            "gpt-5.4",
            "gemini-2.5-pro",
        }
        assert set(CLI_PROXY_MODELS) == expected

    def test_all_entries_are_strings(self) -> None:
        for model in CLI_PROXY_MODELS:
            assert isinstance(model, str)

    def test_no_duplicates(self) -> None:
        assert len(CLI_PROXY_MODELS) == len(set(CLI_PROXY_MODELS))


# ---------------------------------------------------------------------------
# groq_backend factory
# ---------------------------------------------------------------------------


class TestGroqBackendFactory:
    """Verify the groq_backend() convenience factory."""

    def test_returns_openai_compat_backend(self) -> None:
        backend = groq_backend()
        assert isinstance(backend, OpenAICompatBackend)

    def test_default_model_is_llama_70b(self) -> None:
        backend = groq_backend()
        assert backend.model_name == "llama-3.3-70b"

    def test_custom_model(self) -> None:
        backend = groq_backend(model="qwen3-32b")
        assert backend.model_name == "qwen3-32b"

    def test_base_url_is_groq(self) -> None:
        backend = groq_backend()
        assert backend._base_url == "https://api.groq.com/openai/v1"

    def test_explicit_api_key(self) -> None:
        backend = groq_backend(api_key="test-key-123")
        assert backend._api_key == "test-key-123"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "env-key-456")
        backend = groq_backend()
        assert backend._api_key == "env-key-456"

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "env-key")
        backend = groq_backend(api_key="explicit-key")
        assert backend._api_key == "explicit-key"

    def test_missing_key_defaults_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        backend = groq_backend(api_key=None)
        assert backend._api_key == ""


# ---------------------------------------------------------------------------
# CLIProxyBackend
# ---------------------------------------------------------------------------


class TestCLIProxyBackend:
    """Verify CLIProxyBackend defaults and inheritance."""

    def test_is_subclass_of_openai_compat(self) -> None:
        assert issubclass(CLIProxyBackend, OpenAICompatBackend)

    def test_is_subclass_of_llm_backend(self) -> None:
        assert issubclass(CLIProxyBackend, LLMBackend)

    def test_default_model(self) -> None:
        backend = CLIProxyBackend()
        assert backend.model_name == "gemini-3-pro"

    def test_custom_model(self) -> None:
        backend = CLIProxyBackend(model="gpt-5")
        assert backend.model_name == "gpt-5"

    def test_default_base_url(self) -> None:
        backend = CLIProxyBackend()
        assert backend._base_url == "http://localhost:8317/v1"

    def test_default_api_key(self) -> None:
        backend = CLIProxyBackend()
        assert backend._api_key == "jarvis-local-key"

    def test_custom_base_url(self) -> None:
        backend = CLIProxyBackend(base_url="http://custom:9999/v1")
        assert backend._base_url == "http://custom:9999/v1"

    def test_custom_api_key(self) -> None:
        backend = CLIProxyBackend(api_key="other-key")
        assert backend._api_key == "other-key"

    def test_repr_includes_class_name(self) -> None:
        backend = CLIProxyBackend()
        r = repr(backend)
        assert "CLIProxyBackend" in r
        assert "gemini-3-pro" in r

    def test_constants_match_defaults(self) -> None:
        assert CLI_PROXY_BASE_URL == "http://localhost:8317/v1"
        assert CLI_PROXY_API_KEY == "jarvis-local-key"


# ---------------------------------------------------------------------------
# CLIProxyBackend in __init__.py exports
# ---------------------------------------------------------------------------


class TestBackendsExports:
    """Verify the backends package exports CLIProxyBackend."""

    def test_cli_proxy_importable_from_package(self) -> None:
        from cambrian.backends import CLIProxyBackend as Imported
        assert Imported is CLIProxyBackend

    def test_all_contains_cli_proxy(self) -> None:
        import cambrian.backends as backends_pkg
        assert "CLIProxyBackend" in backends_pkg.__all__


# ---------------------------------------------------------------------------
# Router model tier constants
# ---------------------------------------------------------------------------


class TestRouterModelConstants:
    """Verify new model tier constants in router.py."""

    def test_groq_cheap_model_value(self) -> None:
        assert GROQ_CHEAP_MODEL == "llama-3.1-8b"

    def test_groq_medium_model_value(self) -> None:
        assert GROQ_MEDIUM_MODEL == "llama-3.3-70b"

    def test_cli_proxy_premium_model_value(self) -> None:
        assert CLI_PROXY_PREMIUM_MODEL == "gemini-3-pro"

    def test_constants_are_strings(self) -> None:
        assert isinstance(GROQ_CHEAP_MODEL, str)
        assert isinstance(GROQ_MEDIUM_MODEL, str)
        assert isinstance(CLI_PROXY_PREMIUM_MODEL, str)
