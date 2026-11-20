"""Built-ins use explicit tables; packages extend them through one ``qda.plugins`` hook."""

from __future__ import annotations

import importlib
import importlib.metadata as md
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")
Provenance = tuple[str, str | None]
_plugins_loaded = False
_plugins_loading = False
_plugin_failure: Exception | None = None


def load_plugins() -> None:
    """Call installed ``qda.plugins`` registration functions once."""
    global _plugins_loaded, _plugins_loading, _plugin_failure
    if _plugin_failure is not None:
        raise _plugin_failure
    if _plugins_loaded or _plugins_loading:
        return
    _plugins_loading = True
    hook = "discovery"
    try:
        for ep in sorted(md.entry_points(group="qda.plugins"), key=lambda ep: ep.name):
            hook = ep.name
            ep.load()()
        _plugins_loaded = True
    except Exception as exc:
        exc.add_note(f"qda.plugins hook {hook!r} failed")
        _plugin_failure = exc
        raise
    finally:
        _plugins_loading = False


class Registry(Generic[T]):
    """Names mapped to objects or lazy ``module:attribute`` references, sorted on read."""

    def __init__(
        self,
        kind: str,
        entries: Mapping[str, T | str] | None = None,
        *,
        group: str | None = None,
        on_add: Callable[[str, T], None] | None = None,
    ) -> None:
        self.kind, self.group = kind, group or f"qda.{kind}"
        self._entries = dict(entries or {})
        self._on_add = on_add

    def register(self, key: str, obj: T | str) -> T | str:
        """Add a plugin; refuse two different providers of the same name."""
        if key in self._entries:
            old = self._entries[key]
            if old is obj or (isinstance(old, str) and isinstance(obj, str) and old == obj):
                return obj
            if isinstance(obj, str):
                module, attr = obj.split(":")
                obj = getattr(importlib.import_module(module), attr)
            if self[key] is obj:
                return obj
            raise ValueError(
                f"{self.kind} plugin {key!r} is provided twice: by "
                f"{_describe(self[key])} and by {_describe(obj)}"
            )
        if self._on_add is not None and not isinstance(obj, str):
            self._on_add(key, obj)
        self._entries[key] = obj
        return obj

    def __getitem__(self, key: str) -> T:
        load_plugins()
        if key not in self._entries:
            from .step import Refused

            raise Refused(f"unknown {self.kind} plugin {key!r}; known: {list(self)}") from None
        obj = self._entries[key]
        if isinstance(obj, str):
            module, attr = obj.split(":")
            obj = getattr(importlib.import_module(module), attr)
            if self._on_add is not None:
                self._on_add(key, obj)
            self._entries[key] = obj
        return cast(T, obj)

    def __iter__(self) -> Iterator[str]:
        load_plugins()
        return iter(sorted(self._entries))

    def __contains__(self, key: object) -> bool:
        load_plugins()
        return key in self._entries

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def get(self, key: str, default: T | None = None) -> T | None:
        """Return the named plugin, or the default when absent."""
        return self[key] if key in self else default  # noqa: SIM401

    def items(self) -> list[tuple[str, T]]:
        """Return plugin pairs in name order."""
        return [(key, self[key]) for key in self]

    def provenance(self, key: str) -> Provenance:
        """Return the distribution owning the plugin's top-level module."""
        from .output.record import provenance_of

        return provenance_of(self[key])


def _describe(obj: Any) -> str:
    from .output.record import provenance_of

    package, version = provenance_of(obj)
    module = getattr(obj, "__module__", type(obj).__module__)
    name = getattr(obj, "__qualname__", type(obj).__qualname__)
    return f"{package} {version or '(not installed)'} ({module}:{name})"
