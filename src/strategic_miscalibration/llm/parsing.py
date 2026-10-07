"""Recover a structured answer from a model's free text.

Four rungs, tried in order; the one that succeeds is recorded with every answer, because they are not
equally trustworthy:

``whole``    the whole response is the JSON object;
``fenced``   a fenced ```` ```json ```` block;
``brace``    the first balanced ``{...}`` that validates (in a long chain of thought this can be an
             object written while thinking rather than the answer);
``salvage``  fields recovered one by one from malformed or truncated JSON.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from pydantic import BaseModel

__all__ = ["ParseError", "parse"]


class ParseError(ValueError):
    """No rung recovered a valid answer."""


def parse[M: BaseModel](text: str, schema: type[M]) -> tuple[M, str]:
    """Parse ``text`` into ``schema``.

    Returns:
        ``(answer, rung)``.

    Raises:
        ParseError: If nothing validates.
    """
    cleaned = _strip_wrappers(text)
    candidates: list[tuple[str, str]] = []
    if cleaned.strip():
        candidates.append(("whole", cleaned.strip()))
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        candidates.append(("fenced", fenced.group(1)))
    candidates.extend(("brace", obj) for obj in _brace_objects(cleaned))
    for rung, candidate in candidates:
        try:
            return schema.model_validate_json(candidate), rung
        except ValueError:
            continue
    salvaged = _salvage(cleaned, schema)
    if salvaged:
        try:
            return schema.model_validate(salvaged), "salvage"
        except ValueError:
            pass
    raise ParseError(f"no valid {schema.__name__} in response: {text[:300]!r}")


def _strip_wrappers(text: str) -> str:
    """Remove ``<think>`` blocks and harmony-style channel markers that confuse brace matching."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|channel\|>.*?<\|message\|>", " ", text, flags=re.DOTALL)
    return re.sub(r"<\|(?:end|start|return)\|>", " ", text)


def _brace_objects(text: str) -> list[str]:
    """Every top-level ``{...}``, string- and escape-aware; a final unclosed object is kept for salvage."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            ch = text[j]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str and ch in "{}":
                depth += 1 if ch == "{" else -1
                if depth == 0:
                    out.append(text[i : j + 1])
                    break
            j += 1
        else:
            out.append(text[i:])
        i = j + 1
    return out


_VALUE = r'("(?:[^"\\]|\\.)*"?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)'


def _salvage(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Per-field recovery, tolerating a truncated trailing string."""
    out: dict[str, Any] = {}
    for name in schema.model_fields:
        m = re.search(rf'"{re.escape(name)}"\s*:\s*' + _VALUE, text, re.DOTALL)
        if not m:
            continue
        raw = m.group(1)
        if raw.startswith('"'):
            val = raw[1:-1] if raw.endswith('"') and len(raw) > 1 else raw[1:]
            if "\\" in val:
                with contextlib.suppress(UnicodeDecodeError):
                    val = val.encode("utf-8").decode("unicode_escape")
            out[name] = val
        elif raw in ("true", "false"):
            out[name] = raw == "true"
        elif raw == "null":
            out[name] = None
        else:
            out[name] = float(raw) if any(c in raw for c in ".eE") else int(raw)
    return out
