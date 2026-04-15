"""Global pytest configuration for Cambrian test suite.

Sets the WindowsSelectorEventLoopPolicy on Windows to prevent socket buffer
exhaustion (WinError 10055) when many async tests run in the same process.
"""

from __future__ import annotations

import asyncio
import sys


def pytest_configure(config: object) -> None:  # noqa: ARG001
    """Set a stable asyncio event loop policy before any tests run."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
