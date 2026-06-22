# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""Cambrian plugin package.

Plugins extend the core EvolutionEngine with optional bio-inspired behaviours.
Each plugin subclasses :class:`~cambrian.plugins.base.CambrianPlugin`.

Discovery example::

    from cambrian.plugin_registry import PluginRegistry
    registry = PluginRegistry()
    available = registry.discover()  # ['dream', 'quorum', 'tabu', ...]
"""

from cambrian.plugins.base import CambrianPlugin

__all__ = ["CambrianPlugin"]
