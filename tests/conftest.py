"""Global pytest configuration for Cambrian test suite.

On Windows, async event loops should use ProactorEventLoop (the default since
Python 3.8) which does NOT use socket pairs for self-piping — unlike
SelectorEventLoop which drains the socket buffer under heavy test load.

We do NOT set WindowsSelectorEventLoopPolicy here because it would cause
WinError 10055 (socket buffer exhaustion) across the 1500+ test suite.
The targeted SelectorEventLoop fix in TestSpeculate handles the one test
that genuinely requires selector semantics.
"""

from __future__ import annotations
