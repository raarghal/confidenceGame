"""The toy setting: the agent is told its success probability, and the draw is synthetic.

An elicitation is one round-1 decision at an asserted state ``(h, μ, δ)`` and a given draw (easy or
hard): the agent sees ``ρ_t`` and reports. No user is queried; the user the prompt describes is the
game's Bayes-rational user.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.numbers import Number
from ..llm.capabilities import capabilities_for
from ..llm.client import Answer, request
from ..prompts.arms import arm as find_arm
from ..prompts.compose import render
from ..prompts.schemas import ledger_record
from ..theory.game import Game
from .base import Observation, Row, call_columns

__all__ = ["DRAWS", "agent_type_desc", "observations", "prompt_context"]

#: The draws an elicitation cell visits, with the label the data records.
DRAWS = {"easy": "EASY", "hard": "HARD"}


def agent_type_desc(ability: str) -> str:
    """How the prompt names the agent's own type (always strategic; ability as given)."""
    return f"STRATEGIC and {'HIGH' if ability == 'H' else 'LOW'}-ABILITY"


def prompt_context(
    game: Game, h: Number, mu: Number, delta: Number, draw: str, ability: str = "H", round_num: int = 1
) -> dict[str, Any]:
    """Everything the toy blocks read, for one elicitation.

    Game facts (monitoring, a competitor, graded success, the signal values) come from ``game`` so
    that every arm describes the same game.
    """
    t = game.types
    hi, lo = t.abilities[0], t.abilities[1]
    competitor = None
    if game.roster.transparent:
        competitor = {"theta": float(t.p_draw("easy", game.roster.transparent[0].ability))}
    graded = None
    if t.success_depends_on_ability:
        graded = {
            "rho_plus_H": float(t.rho("easy", hi)),
            "rho_plus_L": float(t.rho("easy", lo)),
            "rho_plus_own": float(t.rho("easy", ability)),
        }
    return {
        "round_num": round_num,
        "total_rounds": game.horizon,
        "discount_factor": float(delta),
        "reward": float(game.payoffs.reward),
        "cost": float(game.payoffs.cost),
        "effort": float(game.payoffs.effort),
        "rho_plus": float(t.rho("easy", hi)),
        "rho_minus": float(t.rho("hard", hi)),
        "theta_H": float(t.p_draw("easy", hi)),
        "theta_L": float(t.p_draw("easy", lo)),
        "rho_t": float(t.rho(draw, ability)),
        "signal_low": float(game.reports.values[0]),
        "signal_high": float(game.reports.values[-1]),
        "h": float(h),
        "mu": float(mu),
        "agent_type_desc": agent_type_desc(ability),
        "monitoring": game.monitoring.name,
        "audit_prob": float(game.monitoring.audit_prob),
        "audit_cost": float(game.monitoring.audit_cost),
        "competitor": competitor,
        "graded": graded,
    }


# ------------------------------------------------------------------------------------------------------ observations


def observations(config: Any) -> list[Observation]:
    """Every observation of a toy elicitation run.

    For each arm, state ``(h, μ, δ)`` and draw, ``config.samples_for(arm)`` independent elicitations
    at the configured temperature. Each is one round-1 call; no user is queried.
    """
    game = config.game()
    caps = capabilities_for(config.model)
    out = []
    for arm_name in config.arms:
        a = find_arm("toy", arm_name, config.conjecture)
        for delta in config.grid["delta"]:
            for h in config.grid["h"]:
                for mu in config.grid["mu"]:
                    for draw, label in DRAWS.items():
                        ctx = prompt_context(game, h, mu, delta, draw, config.agent_ability)
                        rendered = render(
                            a, ctx, text_mode=caps.text_mode, answer_only_=caps.text_mode and caps.reasoning_channel
                        )
                        for i in range(config.samples_for(arm_name)):
                            key = f"{arm_name}|{int(config.conjecture)}|{h}|{mu}|{delta}|{draw}|{i}"
                            design = {
                                "arm": arm_name,
                                "conjecture": config.conjecture,
                                "h": h,
                                "mu": mu,
                                "delta": delta,
                                "agent_ability": config.agent_ability,
                                "draw": label,
                                "rho_t": ctx["rho_t"],
                                "sample_idx": i,
                                "model": config.model,
                                "temperature": config.temperature,
                                "monitoring": ctx["monitoring"],
                            }
                            req = request(
                                config.model,
                                rendered.text,
                                rendered.schema,
                                config.temperature,
                                config.max_tokens,
                                key,
                            )
                            out.append(Observation(key, design, (req,), _finisher(ctx)))
    return out


def _finisher(ctx: dict[str, Any]) -> Any:
    def finish(answers: Sequence[Answer]) -> Row:
        (ans,) = answers
        row: Row = {"reasoning": ans.reasoning, **call_columns(ans)}
        p = ans.parsed
        if p is None:
            return {**row, "is_valid": False}
        d = p.model_dump()
        if "signal" in d:
            hi, lo = abs(d["signal"] - ctx["signal_high"]) < 1e-9, abs(d["signal"] - ctx["signal_low"]) < 1e-9
            report_high = True if hi else False if lo else None
            row.update(signal=d["signal"], report_high=report_high, is_valid=report_high is not None)
        if "prob_high" in d:
            row.update(prob_high=min(1.0, max(0.0, d["prob_high"])), is_valid=True)
        if "conj_user_delegate_high" in d:  # a stated probability outside [0, 1] is clipped
            row.update(
                conj_delegate_high=min(1.0, max(0.0, d["conj_user_delegate_high"])),
                conj_delegate_low=min(1.0, max(0.0, d["conj_user_delegate_low"])),
            )
        row.update(ledger_record(p))  # the ledger is recorded raw, never clipped
        return row

    return finish
