# Copyright 2026 Cambrian Authors. SPDX-License-Identifier: MIT
"""PluginRegistry -- manages plugin discovery, loading, and lifecycle."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cambrian.plugins.base import CambrianPlugin
from cambrian.utils.logging import get_logger

if TYPE_CHECKING:
    from cambrian.evolution import EvolutionEngine

logger = get_logger(__name__)

_RESERVED_NAMES = {"base"}


class PluginRegistry:
    """Discovers, loads, and manages the lifecycle of Cambrian plugins.

    Usage::

        from cambrian.plugin_registry import PluginRegistry

        registry = PluginRegistry()

        # List available plugins
        for meta in registry.list_all():
            print(meta["name"], "-", meta["description"])

        # Enable plugins for an evolution run
        engine = EvolutionEngine(...)
        registry.enable(["dream", "quorum"], engine)

        # Load from YAML config
        registry.load_from_yaml("cambrian_config.yaml", engine)
    """

    def __init__(self) -> None:
        self._loaded: dict[str, CambrianPlugin] = {}

    # -- Discovery -------------------------------------------------------------

    def discover(self) -> list[str]:
        """Return names of all available plugin modules in ``cambrian/plugins/``.

        Returns:
            Sorted list of plugin names (e.g. ``["annealing", "dream", ...]``).
        """
        import cambrian.plugins as _pkg

        plugins_path = Path(_pkg.__file__).parent
        names = sorted(
            info.name
            for info in pkgutil.iter_modules([str(plugins_path)])
            if info.name not in _RESERVED_NAMES
        )
        return names

    # -- Loading ---------------------------------------------------------------

    def load(self, name: str) -> CambrianPlugin:
        """Load and return a plugin instance by *name*.

        The plugin module ``cambrian.plugins.<name>`` must define exactly one
        :class:`~cambrian.plugins.base.CambrianPlugin` subclass with
        ``plugin.name == name``.

        Args:
            name: Plugin identifier (e.g. ``"dream"``).

        Returns:
            A :class:`~cambrian.plugins.base.CambrianPlugin` instance.

        Raises:
            ModuleNotFoundError: If the module does not exist.
            ValueError: If no matching plugin class is found in the module.
        """
        if name in self._loaded:
            return self._loaded[name]

        module = importlib.import_module(f"cambrian.plugins.{name}")

        for attr_name in dir(module):
            attr = getattr(module, attr_name, None)
            if (
                isinstance(attr, type)
                and issubclass(attr, CambrianPlugin)
                and attr is not CambrianPlugin
                and getattr(attr, "name", "") == name
            ):
                instance = attr()
                self._loaded[name] = instance
                logger.debug("Loaded plugin %r (%s)", name, type(instance).__name__)
                return instance

        raise ValueError(
            f"No CambrianPlugin with name={name!r} found in cambrian.plugins.{name}. "
            "Ensure the plugin class has `name = <plugin_name>` set."
        )

    # -- Enabling / Disabling --------------------------------------------------

    def enable(self, names: list[str], engine: "EvolutionEngine") -> None:
        """Enable *names* plugins: load, register hooks, activate.

        Args:
            names: Plugin identifiers to enable.
            engine: :class:`~cambrian.evolution.EvolutionEngine` to register
                hooks on.
        """
        for name in names:
            plugin = self.load(name)
            plugin.register(engine)
            plugin.activate()
            logger.info("Plugin %r enabled", name)

    def disable(self, name: str) -> None:
        """Deactivate a loaded plugin.

        Args:
            name: Plugin identifier.
        """
        if name in self._loaded:
            self._loaded[name].deactivate()
            logger.info("Plugin %r disabled", name)

    # -- YAML config -----------------------------------------------------------

    def load_from_yaml(self, config_path: str, engine: "EvolutionEngine") -> None:
        """Load and enable plugins listed in a YAML config file.

        Expected YAML format::

            plugins:
              enabled:
                - dream
                - quorum
                - tabu

        Args:
            config_path: Path to YAML configuration file.
            engine: EvolutionEngine instance to register hooks on.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If *config_path* does not exist.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config loading: pip install pyyaml"
            ) from exc

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_file.open() as fh:
            config: dict[str, Any] = yaml.safe_load(fh) or {}

        enabled: list[str] = (config.get("plugins") or {}).get("enabled") or []
        if enabled:
            self.enable(enabled, engine)
            logger.info("Loaded %d plugins from %s", len(enabled), config_path)
        else:
            logger.debug("No plugins.enabled in %s", config_path)

    # -- Introspection ---------------------------------------------------------

    @property
    def active_plugins(self) -> list[str]:
        """Names of currently active plugins."""
        return [name for name, p in self._loaded.items() if p.is_active]

    def list_all(self) -> list[dict[str, str | None]]:
        """Return metadata for all discoverable plugins.

        Plugins that fail to load will appear with ``description="(load error)"``.

        Returns:
            List of metadata dicts (keys: name, description, category, impact).
        """
        result: list[dict[str, str | None]] = []
        for name in self.discover():
            try:
                plugin = self.load(name)
                result.append(plugin.metadata())
            except Exception as exc:  # noqa: BLE001
                result.append(
                    {
                        "name": name,
                        "description": f"(load error: {exc})",
                        "category": "unknown",
                        "impact": None,
                    }
                )
        return result

    def __repr__(self) -> str:
        return (
            f"PluginRegistry(loaded={list(self._loaded)!r}, "
            f"active={self.active_plugins!r})"
        )
