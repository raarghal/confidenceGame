"""Every prompt is five blocks: ``role``, ``setup``, ``state``, ``query``, ``output``.

Each block is one section of the prompt and prints its own heading, so a prompt is its five blocks
joined by blank lines. Every prompt has all five. A prompt *variant* changes the text of one or more
blocks, and nothing else:

=========  ============================================================  =========================================
block      content                                                       variants (``templates/blocks/<block>/``)
=========  ============================================================  =========================================
role       who the agent is, what it is paid, how the rounds are weighed  signal, strategy, signal_paid_on_success,
                                                                         streamlined, neutral
setup      the game: the user, payoffs, monitoring, the private draw,    toy, toy_semantics, math_qa,
           the types (and a competitor or graded success, if any)        math_qa_paid_on_success, streamlined, neutral
state      the current state: ρ (toy), the beliefs, the question         toy_action, toy_action_link, toy_strategy,
                                                                         math_qa, streamlined, task_only
query      the instruction, including the length budget                  see ``schemas.QUERY_SCHEMAS``
output     the JSON answer format                                        generated from the query's schema
=========  ============================================================  =========================================

An :class:`Arm` picks a variant for each of the first four blocks; ``output`` follows from the query.
Derive an arm by changing one block: ``CUED = LINK.with_blocks("cued", query="cued")``.

Facts about the *game* are not variants. Monitoring, a competitor and graded success are rendered from the
game through the context, so a generalization of the game reaches every arm with no new variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from .output_format import json_instructions
from .schemas import answer_only, schema_for

__all__ = ["BLOCKS", "Arm", "Rendered", "render", "render_blocks", "render_instrument", "variants"]

TEMPLATES = Path(__file__).parent / "templates"
BLOCKS = ("role", "setup", "state", "query")  # the fifth, ``output``, is generated from the query

_ENV = Environment(
    loader=FileSystemLoader([TEMPLATES / "blocks", TEMPLATES]),
    undefined=StrictUndefined,  # a block that uses a variable the context lacks fails loudly
    autoescape=False,
)


@dataclass(frozen=True)
class Arm:
    """A prompt arm: one text variant for each block.

    Attributes:
        name: The name recorded in every row the arm produces.
        setting: ``"toy"`` or ``"math_qa"``; the setting supplies the render context.
        blocks: Variant name for each of ``role``, ``setup``, ``state``, ``query``.
        leading_blank_line: Historical: four of the paper's toy prompts began with a blank line (a
            template comment's newline). Kept so they render exactly as sent.
        doc: What the arm isolates and why it exists.
    """

    name: str
    setting: Literal["toy", "math_qa"]
    blocks: Mapping[str, str]
    leading_blank_line: bool = False
    doc: str = ""

    def __post_init__(self) -> None:
        if set(self.blocks) != set(BLOCKS):
            raise ValueError(f"{self.name}: blocks must name exactly {BLOCKS}, got {sorted(self.blocks)}")
        for block, variant in self.blocks.items():
            if not (TEMPLATES / "blocks" / block / f"{variant}.j2").is_file():
                raise ValueError(f"{self.name}: no {block} variant {variant!r}")
        object.__setattr__(self, "blocks", MappingProxyType(dict(self.blocks)))

    def with_blocks(self, name: str, doc: str = "", leading_blank_line: bool | None = None, **blocks: str) -> Arm:
        """A new arm with the text of some blocks replaced."""
        blank = self.leading_blank_line if leading_blank_line is None else leading_blank_line
        return replace(self, name=name, doc=doc, blocks={**self.blocks, **blocks}, leading_blank_line=blank)

    def schema(self, context: Mapping[str, Any]) -> type[BaseModel]:
        """The response schema this arm's query asks for."""
        return schema_for(self.blocks["query"], context)


@dataclass(frozen=True)
class Rendered:
    """A rendered prompt and the schema its answer must satisfy."""

    text: str
    schema: type[BaseModel]


def render(arm: Arm, context: Mapping[str, Any], *, text_mode: bool = True, answer_only_: bool = False) -> Rendered:
    """Render ``arm`` in ``context``.

    Args:
        arm: The arm.
        context: Values the blocks read (the setting builds it).
        text_mode: Append the JSON skeleton (for models asked for plain text); otherwise the provider
            enforces the schema and the output block is empty.
        answer_only_: Drop the ``reasoning`` field (for models whose reasoning has its own channel).

    Returns:
        The prompt and its schema.
    """
    blocks, schema = render_blocks(arm, context, text_mode=text_mode, answer_only_=answer_only_)
    text = ("\n" if arm.leading_blank_line else "") + "\n\n".join(blocks.values())
    return Rendered(text, schema)


def render_blocks(
    arm: Arm, context: Mapping[str, Any], *, text_mode: bool = True, answer_only_: bool = False
) -> tuple[dict[str, str], type[BaseModel]]:
    """Each of the five blocks rendered separately, in prompt order, and the answer schema."""
    schema = arm.schema(context)
    if answer_only_:
        schema = answer_only(schema)
    blocks = {b: _ENV.get_template(f"{b}/{arm.blocks[b]}.j2").render(context) for b in BLOCKS}
    blocks["output"] = json_instructions(schema) if text_mode else ""
    return blocks, schema


def render_instrument(name: str, context: Mapping[str, Any]) -> str:
    """Render a fixed instrument prompt (``instruments/<name>.j2``), e.g. the neutral baseline for ``ρ``."""
    return _ENV.get_template(f"instruments/{name}.j2").render(context)


def variants(block: str) -> list[str]:
    """The text variants available for ``block``."""
    return sorted(p.stem for p in (TEMPLATES / "blocks" / block).glob("*.j2"))
