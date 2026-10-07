"""One registry type for every extension point.

Solvers, prompt arms and settings are all looked up by name. Each of those
modules owns one :class:`Registry` instance; this class is the only implementation, so lookup errors
read the same everywhere and list what *is* registered.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

__all__ = ["Registry"]


class Registry[T]:
    """A name -> object map that refuses silent overwrites.

    Args:
        kind: What is being registered, used in error messages (e.g. ``"prompt arm"``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, item: T) -> T:
        """Register ``item`` under ``name``.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._items:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item
        return item

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator form of :meth:`add`."""

        def decorate(item: T) -> T:
            return self.add(name, item)

        return decorate

    def __getitem__(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {known}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[tuple[str, T]]:
        """Registered ``(name, item)`` pairs in registration order."""
        return list(self._items.items())
