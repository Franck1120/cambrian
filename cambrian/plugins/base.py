# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Base class for all Cambrian plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cambrian.evolution import EvolutionEngine


class CambrianPlugin(ABC):
    """Base class for all Cambrian plugins.

    Each plugin exposes metadata (name, description, category, impact) and
    implements register() to optionally attach hooks to an EvolutionEngine.

    Lifecycle::

        registry = PluginRegistry()
        registry.enable(["dream", "quorum"], engine)
        # Each plugin: register() called -> activate() called

    Implementing a plugin::

        class MyPlugin(CambrianPlugin):
            name = "my_plugin"
            description = "Does something useful."
            category = "evaluation"
            impact = "+5% fitness on benchmark X"

            def register(self, engine: EvolutionEngine) -> None:
                engine.add_hook("on_generation_end", self._my_hook)

            def _my_hook(self, generation: int, population: list) -> None:
                pass
    """

    #: Unique plugin identifier (e.g. ``"dream"``, ``"quorum"``).
    name: str = ""
    #: Human-readable description of what the plugin does.
    description: str = ""
    #: Functional category (e.g. ``"memory"``, ``"selection"``, ``"evaluation"``).
    category: str = ""
    #: Measured or estimated performance impact, if available.
    impact: str | None = None

    def __init__(self) -> None:
        self._active = False

    @abstractmethod
    def register(self, engine: "EvolutionEngine") -> None:
        """Register hooks with *engine*.

        Override to attach callables via ``engine.add_hook(hook_name, fn)``.
        Leave as a no-op if the plugin is used standalone (not engine-integrated).
        """

    def activate(self) -> None:
        """Mark the plugin as active."""
        self._active = True

    def deactivate(self) -> None:
        """Mark the plugin as inactive."""
        self._active = False

    @property
    def is_active(self) -> bool:
        """Whether the plugin is currently active."""
        return self._active

    def metadata(self) -> dict[str, str | None]:
        """Return plugin metadata as a dict."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "impact": self.impact,
        }

    def __repr__(self) -> str:
        state = "active" if self._active else "inactive"
        return f"{type(self).__name__}(name={self.name!r}, {state})"
