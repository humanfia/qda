"""Deprecated: the job sandbox's bwrap code moved to `qda_agent.isolation.bwrap`.

The names this module had are still reachable here for one release, with a `DeprecationWarning` on import, so
existing importers keep working. Every read and every write goes to `qda_agent.isolation.bwrap`, so a caller
that adjusts a module global (`_ns`, `on_skip`, `NV_PROC`, a helper override) on this path changes the one
implementation and reads back what it set.
"""

from __future__ import annotations

import sys as _sys
import types as _types
import warnings as _warnings
from typing import TYPE_CHECKING, Any

from qda_agent.isolation import bwrap as _bwrap

if TYPE_CHECKING:  # the static view type checkers read; at run time every name is forwarded below
    from qda_agent.isolation.bwrap import *  # noqa: F403
    from qda_agent.isolation.bwrap import _ns, _probe_ns  # noqa: F401 -- named only by callers' tests

_warnings.warn(
    "qda_agent.sched.sandbox moved to qda_agent.isolation.bwrap; the old import path is deprecated and will "
    "be removed in a later release",
    DeprecationWarning,
    stacklevel=2,
)

#: Names this module owns; every other attribute is the implementation's.
_LOCAL = frozenset({"_LOCAL", "_bwrap", "_sys", "_types", "_warnings", "_Forwarding", "__getattr__"})


def __getattr__(name: str) -> Any:
    """Resolve a name the new module gained after this one was imported."""
    return getattr(_bwrap, name)


class _Forwarding(_types.ModuleType):
    """Read and write every implementation name through `qda_agent.isolation.bwrap`.

    The names are also copied into this module's `__dict__` so `pytest.monkeypatch` restores what it found;
    reads never use those copies, because the implementation itself advances a global (`_ns`) without this
    module seeing the assignment.
    """

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__") or name in _LOCAL:
            return super().__getattribute__(name)
        return getattr(_bwrap, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("__"):
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
            setattr(_bwrap, name, value)

    def __delattr__(self, name: str) -> None:
        """Undo a write on this path (`pytest.monkeypatch` restores by deleting a name it added).

        The module's own copy goes as well as the implementation's: a copy left in this module's `__dict__`
        is what a later reader (or a second patch's undo) would resurrect.
        """
        if name.startswith("__"):
            super().__delattr__(name)
            return
        try:
            super().__delattr__(name)
        except AttributeError:
            pass
        try:
            delattr(_bwrap, name)
        except AttributeError:
            pass


#: Every name of the new module, by identity (for `__dict__` readers such as `pytest.monkeypatch`).
globals().update({name: value for name, value in vars(_bwrap).items() if not name.startswith("__")})
_sys.modules[__name__].__class__ = _Forwarding
