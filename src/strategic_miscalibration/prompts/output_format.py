"""The output-format block: a JSON skeleton generated from a response schema.

Used when a model is asked for plain text rather than for natively enforced JSON. The skeleton shows
field names and value shapes only (descriptions are not shown), so for those models the schema's names
and order are the entire specification of the answer. Arrays show ``minItems`` exemplars, so a
``min_length=2`` constraint is visible as two elements.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["json_instructions"]

_PLACEHOLDERS = {"string": '"<string>"', "number": "<float>", "integer": "<integer>", "boolean": "<true/false>"}
_MAX_EXEMPLARS = 3


def json_instructions(schema: type[BaseModel]) -> str:
    """``Respond with ONLY the following JSON object …`` followed by the schema's skeleton."""
    js = schema.model_json_schema()
    defs = js.get("$defs", {})
    body = ",\n".join(f'  "{name}": {_value(spec, defs, "  ")}' for name, spec in js.get("properties", {}).items())
    return f"Respond with ONLY the following JSON object (no markdown, no extra text):\n{{\n{body}\n}}"


def _resolve(spec: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    ref = spec.get("$ref", "")
    resolved: dict[str, Any] = defs.get(ref.split("/")[-1], spec) if ref.startswith("#/$defs/") else spec
    return resolved


def _value(spec: dict[str, Any], defs: dict[str, Any], indent: str) -> str:
    spec = _resolve(spec, defs)
    kind = spec.get("type")
    if kind == "object":
        props = spec.get("properties", {})
        if not props:
            return "{...}"
        rendered = {n: _value(s, defs, indent + "  ") for n, s in props.items()}
        if all("\n" not in v for v in rendered.values()):  # an all-scalar object reads best on one line
            return "{" + ", ".join(f'"{n}": {v}' for n, v in rendered.items()) + "}"
        inner = ",\n".join(f'{indent}  "{n}": {v}' for n, v in rendered.items())
        return f"{{\n{inner}\n{indent}}}"
    if kind == "array":
        item = _value(spec.get("items", {}), defs, indent + "  ")
        n = max(1, min(int(spec.get("minItems") or 1), _MAX_EXEMPLARS))
        lines = [f"{indent}  {item}," for _ in range(n)]
        if spec.get("maxItems") != n:
            lines.append(f"{indent}  ...")
        else:
            lines[-1] = lines[-1].rstrip(",")
        return "[\n" + "\n".join(lines) + f"\n{indent}]"
    return _PLACEHOLDERS.get(kind or "", "<value>")
