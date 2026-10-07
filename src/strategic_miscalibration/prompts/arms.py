"""Every prompt arm the paper's runs used, written as derivations from one base per setting.

Read top to bottom this file *is* the paper's ablation structure: each arm differs from the arm it is
derived from in exactly the blocks named in its ``with_blocks`` call. Arm names are the ``version``
values recorded in the data. The clauses the toy arms vary are C1 (the user's decision rule, in
``setup``), C2 (the draw is private) and C3 (a strategic type's signal need not match the truth, in
either direction).

Four toy arms were run from templates whose comment header left a blank first line; that line is kept
(``leading_blank_line``) so every arm renders exactly as it was sent.
"""

from __future__ import annotations

from ..core.registry import Registry
from .compose import Arm

__all__ = ["MATH_QA_ARMS", "TOY_ARMS", "arm", "conjecture_variant"]

TOY_ARMS: Registry[Arm] = Registry("toy prompt arm")
MATH_QA_ARMS: Registry[Arm] = Registry("math Q&A prompt arm")


def _toy(a: Arm) -> Arm:
    return TOY_ARMS.add(a.name, a)


def _mq(a: Arm) -> Arm:
    return MATH_QA_ARMS.add(a.name, a)


# --------------------------------------------------------------------------------------------------------------- toy

CLARIFIED = _toy(
    Arm(
        "minimal_clarified",
        "toy",
        {"role": "signal", "setup": "toy", "state": "toy_action", "query": "action"},
        doc="Clauses C1, C2 and C3.",
    )
)
SEMANTICS = _toy(
    CLARIFIED.with_blocks(
        "minimal_semantics",
        setup="toy_semantics",
        leading_blank_line=True,
        doc="C2 and C3 without C1: the counterpart's objective is not stated.",
    )
)
LINK = _toy(
    CLARIFIED.with_blocks(
        "minimal_clarified_link",
        state="toy_action_link",
        leading_blank_line=True,
        doc="Adds the sentence that the user's beliefs carry over to the next round.",
    )
)
CUED = _toy(
    LINK.with_blocks(
        "minimal_clarified_cued",
        query="cued",
        doc="Adds an instruction to work out this round's decision and next round's beliefs before choosing.",
    )
)
LEDGER = _toy(
    CUED.with_blocks(
        "minimal_clarified_ledger",
        query="ledger",
        doc="Demands the four round payoffs (this/next round x HIGH/LOW) before the signal, without a length cap. "
        "The paper's main arm.",
    )
)
STRATEGY = _toy(
    CLARIFIED.with_blocks(
        "strategy_clarified",
        role="strategy",
        state="toy_strategy",
        query="strategy",
        doc="minimal_clarified asking for the probability of reporting HIGH instead of a report.",
    )
)
_toy(
    STRATEGY.with_blocks(
        "strategy_semantics",
        setup="toy_semantics",
        leading_blank_line=True,
        doc="strategy_clarified without C1.",
    )
)

# ---------------------------------------------------------------------------------------------------------- math Q&A

MQ_CUED = _mq(
    Arm(
        "minimal_clarified_cued",
        "math_qa",
        {"role": "signal", "setup": "math_qa", "state": "math_qa", "query": "mq_cued"},
        doc="The toy cued arm with no stated rho (it is measured) and ability described in words. The control "
        "the ledger arm is read against.",
    )
)
_mq(
    MQ_CUED.with_blocks(
        "minimal_clarified_ledger",
        query="mq_ledger",
        doc="Names and prices at least two candidate reports before reporting, without a length cap.",
    )
)
_mq(
    MQ_CUED.with_blocks(
        "minimal_clarified_caponly",
        query="mq_caponly",
        doc="Cued with the length cap lifted: separates the cap from the ledger's demand.",
    )
)
_mq(
    MQ_CUED.with_blocks(
        "minimal_clarified_noreason",
        query="mq_plain",
        doc="Cued without the instruction to reason: separates reasoning load from the stakes.",
    )
)
_mq(
    MQ_CUED.with_blocks(
        "minimal_clarified_paid_on_success",
        role="signal_paid_on_success",
        setup="math_qa_paid_on_success",
        doc="Paid only when correct. A different game: no equilibrium, threshold or welfare number applies.",
    )
)
_mq(
    MQ_CUED.with_blocks(
        "minimal_clarified_streamlined",
        role="streamlined",
        setup="streamlined",
        state="streamlined",
        query="mq_streamlined",
        doc="The cued game in about half the words: a prompt-length control.",
    )
)
_mq(
    MQ_CUED.with_blocks(
        "neutral_padded",
        role="neutral",
        setup="neutral",
        state="task_only",
        query="mq_neutral",
        doc="No game, at the game prompt's length: the pure length effect.",
    )
)

_CONJECTURE = {"action": "conjecture", "strategy": "strategy_conjecture"}


def conjecture_variant(base: Arm) -> Arm:
    """The arm with the agent first stating its conjecture about the user's response (a query variant).

    Raises:
        ValueError: For arms that deliberately have none: eliciting the conjecture in the cued or ledger
            arm would reintroduce the static object those arms replace.
    """
    query = base.blocks["query"]
    if query not in _CONJECTURE:
        raise ValueError(f"{base.name} has no conjecture variant")
    return base.with_blocks(base.name, doc=f"{base.doc} Conjecture elicited.", query=_CONJECTURE[query])


def arm(setting: str, name: str, conjecture: bool = False) -> Arm:
    """Look up an arm by setting and name, optionally its conjecture variant."""
    found = (TOY_ARMS if setting == "toy" else MATH_QA_ARMS)[name]
    return conjecture_variant(found) if conjecture else found
