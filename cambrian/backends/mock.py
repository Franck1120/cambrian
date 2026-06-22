# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""MockBackend — an offline, deterministic stand-in for a real LLM.

This backend performs **no inference**. It exists so the evolutionary loop,
the API server, and the benchmark harness can run end-to-end without an API
key or network access — for plumbing tests, demos, and CI.

Honesty contract
----------------
Output from this backend is NOT model-generated text. ``generate`` returns a
deterministic, hash-derived string. Any fitness measured against it reflects
the *search machinery*, not real LLM capability. Callers MUST surface to the
user that results came from the mock (the API tags runs ``backend="mock"``;
the benchmark labels its output ``mock``). Never present mock output as if a
real model produced it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cambrian.backends.base import LLMBackend


class MockBackend(LLMBackend):
    """Deterministic offline backend. Does no real inference.

    For genome-mutation prompts (the LLMMutator sends JSON), ``generate``
    echoes back a valid genome JSON with a slightly perturbed temperature, so
    the mutation/crossover plumbing produces parseable output. For any other
    prompt it returns a short deterministic acknowledgement.

    Args:
        model: Reported model name. Default ``"mock"``.
    """

    def __init__(self, model: str = "mock") -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # If the prompt embeds a genome JSON (mutator/crossover path), return a
        # parseable genome so the loop continues. Otherwise return a stub reply.
        start = prompt.find("{")
        end = prompt.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(prompt[start : end + 1])
                if isinstance(data, dict) and "system_prompt" in data:
                    digest = hashlib.sha256(prompt.encode()).hexdigest()
                    nudge = (int(digest[:4], 16) % 21 - 10) / 100.0  # [-0.10, 0.10]
                    base_temp = float(data.get("temperature", 0.7))
                    data["temperature"] = max(0.1, min(1.5, base_temp + nudge))
                    return json.dumps(data)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock:{self._model}] deterministic reply {digest}"

    def __repr__(self) -> str:
        return f"MockBackend(model={self._model!r})"
