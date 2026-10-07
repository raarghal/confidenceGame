"""Ask a model for a structured answer: one call, or many in parallel.

:meth:`Client.ask` never raises. A failed call is an observation with a recorded reason
(``failure = "api"`` or ``"parse"``) and, where the model answered, its raw text, so every failure
can be diagnosed from the data rather than from a log.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .backends import Backend, Completion, Request
from .capabilities import capabilities_for
from .parsing import ParseError, parse

__all__ = ["Answer", "Client", "request"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Answer:
    """The outcome of one request.

    Attributes:
        parsed: The validated answer, or ``None`` on failure.
        raw: The response text exactly as returned (empty if the call itself failed).
        reasoning: The chain of thought: the reasoning channel if the model has one, else the answer's
            own ``reasoning`` field.
        rung: How the answer was recovered (``schema`` when the provider enforced it; see
            :mod:`.parsing`), or ``failed``.
        cost: USD for this call (0 when served from the cache).
        cached: Served from the cache.
        failure: ``None``, ``"api"`` (the call failed after retries) or ``"parse"`` (no valid answer).
        error: The failure message.
    """

    parsed: BaseModel | None
    raw: str
    reasoning: str
    rung: str
    cost: float
    cached: bool
    failure: Literal["api", "parse"] | None = None
    error: str = ""


def request(
    model: str, prompt: str, schema: type[BaseModel], temperature: float, max_tokens: int | None = None, key: str = ""
) -> Request:
    """A :class:`Request` with the model's own output route and default budget."""
    caps = capabilities_for(model)
    return Request(model, prompt, schema, temperature, max_tokens or caps.max_tokens, caps.text_mode, key)


class Client:
    """Sends requests through a backend and tracks spend.

    Args:
        backend: Where completions come from.
        max_workers: Parallel requests in :meth:`ask_all`.
    """

    def __init__(self, backend: Backend, max_workers: int = 8) -> None:
        self.backend = backend
        self.max_workers = max_workers
        self.spent = 0.0
        self._lock = threading.Lock()

    def ask(self, req: Request) -> Answer:
        """One request; failures are returned, not raised."""
        try:
            done: Completion = self.backend.complete(req)
        except Exception as e:  # noqa: BLE001 -- any provider error is recorded as an API failure
            logger.warning("call failed (%s): %s", req.key, e)
            return Answer(None, "", "", "failed", 0.0, False, "api", str(e))
        with self._lock:
            self.spent += done.cost
        try:
            if req.text_mode:
                parsed, rung = parse(done.content, req.schema)
            else:
                parsed, rung = req.schema.model_validate_json(done.content), "schema"
        except (ParseError, ValueError) as e:
            return Answer(None, done.content, done.reasoning, "failed", done.cost, done.cached, "parse", str(e))
        reasoning = done.reasoning or str(getattr(parsed, "reasoning", "") or "")
        return Answer(parsed, done.content, reasoning, rung, done.cost, done.cached)

    def ask_all(self, requests: Sequence[Request]) -> list[Answer]:
        """Many requests in parallel; results in request order."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.ask, requests))
