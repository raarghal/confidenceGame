"""Where completions come from. Every backend has one method, ``complete(request) -> Completion``.

- :class:`LiteLLMBackend` — a live provider through LiteLLM (needs the ``llm`` extra and an API key).
- :class:`FakeBackend` — deterministic answers from a function of the request, for tests and dry
  runs; no network.
- :class:`CachedBackend` — wraps another backend and stores every completion on disk, so rerunning
  a finished or interrupted run costs nothing.

A replay backend serving recorded responses would be one more class with the same method.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .capabilities import capabilities_for

__all__ = ["Backend", "CachedBackend", "Completion", "FakeBackend", "LiteLLMBackend", "Request", "fill_schema"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    """One model call.

    Attributes:
        model: Provider model string, e.g. ``together_ai/openai/gpt-oss-120b``.
        prompt: The full user message.
        schema: The answer's schema.
        temperature: Sampling temperature.
        max_tokens: Token budget.
        text_mode: Plain text (schema in the prompt) rather than native schema enforcement.
        key: What makes this call distinct from an identical one, e.g. the sample index of a repeated
            elicitation; part of the cache key, so repeated samples are not served from one cache entry.
    """

    model: str
    prompt: str
    schema: type[BaseModel]
    temperature: float
    max_tokens: int
    text_mode: bool
    key: str = ""

    def fingerprint(self) -> str:
        """A stable hash of everything that determines the completion."""
        blob = json.dumps(
            [
                self.model,
                self.prompt,
                self.schema.model_json_schema(),
                self.temperature,
                self.max_tokens,
                self.text_mode,
                capabilities_for(self.model).reasoning_effort,
                self.key,
            ],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class Completion:
    """What a backend returned: the answer text, the reasoning-channel text, and the cost in USD."""

    content: str
    reasoning: str = ""
    cost: float = 0.0
    cached: bool = False


class Backend(Protocol):
    """A source of completions."""

    def complete(self, request: Request) -> Completion: ...


class EmptyResponse(RuntimeError):
    """The provider returned no content (retried)."""


class LiteLLMBackend:
    """A live provider through LiteLLM, retrying transient failures with exponential backoff.

    Args:
        attempts: Tries per request before the failure is recorded.
    """

    def __init__(self, attempts: int = 5) -> None:
        import litellm
        from tenacity import retry, stop_after_attempt, wait_random_exponential

        from ..core.env import load_secrets

        load_secrets()
        self._litellm = litellm
        self._call = retry(
            stop=stop_after_attempt(attempts), wait=wait_random_exponential(min=5, max=120), reraise=True
        )(self._once)

    def complete(self, request: Request) -> Completion:
        result: Completion = self._call(request)
        return result

    def _once(self, request: Request) -> Completion:
        caps = capabilities_for(request.model)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        allowed = []
        if not request.text_mode:
            kwargs["response_format"] = {"type": "json_schema", "schema": request.schema.model_json_schema()}
            if not self._knows(request.model):
                allowed.append("response_format")  # LiteLLM refuses locally for models it has not mapped
        if caps.reasoning_effort is not None:
            kwargs["reasoning_effort"] = caps.reasoning_effort
            allowed.append("reasoning_effort")
        if allowed:
            kwargs["allowed_openai_params"] = allowed
        response = self._litellm.completion(**kwargs)
        try:
            cost = float(self._litellm.completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001 -- an unpriced model is not a failed call
            cost = 0.0
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning_content", "") or ""
        content = message.content or reasoning  # a reasoning model may leave the answer in its reasoning
        if not content:
            raise EmptyResponse("empty response")
        return Completion(content, reasoning, cost)

    def _knows(self, model: str) -> bool:
        try:
            self._litellm.get_model_info(model)
        except Exception:  # noqa: BLE001
            return False
        return True


class FakeBackend:
    """Deterministic, offline completions.

    Args:
        respond: ``request -> answer dict``. The default fills the schema with values derived from the
            request's fingerprint, so a run is reproducible and every field is valid.
    """

    def __init__(self, respond: Callable[[Request], dict[str, Any]] | None = None) -> None:
        self.respond = respond or (lambda r: fill_schema(r.schema, random.Random(r.fingerprint())))
        self.calls = 0

    def complete(self, request: Request) -> Completion:
        self.calls += 1
        return Completion(json.dumps(self.respond(request)), reasoning="(fake)")


def fill_schema(schema: type[BaseModel], rng: random.Random) -> dict[str, Any]:
    """A valid instance of ``schema`` as a dict: strings ``"A"``, numbers in ``[0, 1]``, lists of two."""
    out: dict[str, Any] = {}
    for name, f in schema.model_fields.items():
        ann = f.annotation
        if ann is str:
            out[name] = "A"
        elif ann in (float, int):
            out[name] = round(rng.random(), 3)
        elif getattr(ann, "__origin__", None) is list:
            (item,) = ann.__args__  # type: ignore[union-attr]
            out[name] = [fill_schema(item, rng) for _ in range(2)]
        else:
            raise TypeError(f"cannot fill field {name!r} of type {ann}")
    return out


class CachedBackend:
    """Stores every completion under its request fingerprint, and serves it again for free."""

    def __init__(self, inner: Backend, directory: Path) -> None:
        self.inner = inner
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def complete(self, request: Request) -> Completion:
        path = self.directory / f"{request.fingerprint()}.json"
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            return Completion(stored["content"], stored["reasoning"], 0.0, cached=True)
        done = self.inner.complete(request)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"content": done.content, "reasoning": done.reasoning, "cost": done.cost}))
        tmp.replace(path)  # atomic: a killed run never leaves a half-written entry
        return done
