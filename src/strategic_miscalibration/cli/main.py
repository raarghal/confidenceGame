"""``smc``: the command line.

smc run CONFIG [--out DIR] [--dry-run] [--workers N] [--cache DIR]
smc prompt SETTING ARM [--conjecture] [--h H --mu MU --delta D --draw easy|hard]
smc prompt-diff SETTING ARM_A ARM_B
smc arms
"""

from __future__ import annotations

import argparse
import difflib
import logging
from pathlib import Path

from ..core.env import output_dir
from ..elicit.config import load
from ..elicit.runner import plan, run
from ..llm.backends import Backend, CachedBackend
from ..llm.client import Client
from ..prompts.arms import MATH_QA_ARMS, TOY_ARMS, arm
from ..prompts.compose import BLOCKS, render_blocks
from ..settings import math_qa, toy
from ..theory.game import paper_game

EXAMPLE_TASK = "Example question?\n\n(A) 1\n(B) 2\n(C) 3"


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    corpus_rows = math_qa.CORPORA[cfg.corpus].load() if cfg.setting == "math_qa" else None
    print(plan(cfg, corpus_rows))
    if args.dry_run:
        return 0
    from ..llm.backends import LiteLLMBackend

    backend: Backend = LiteLLMBackend()
    if args.cache:
        backend = CachedBackend(backend, args.cache)
    out = args.out or output_dir() / "runs" / cfg.name
    table = run(cfg, args.config, out, Client(backend, args.workers), corpus_rows)
    print(f"{len(table):,} rows ({int(table.is_valid.sum()):,} valid) -> {out}")
    return 0


def _blocks(setting: str, name: str, args: argparse.Namespace) -> dict[str, str]:
    a = arm(setting, name, getattr(args, "conjecture", False))
    if setting == "toy":
        ctx = toy.prompt_context(paper_game(), args.h, args.mu, args.delta, args.draw)
    else:
        ctx = math_qa.prompt_context(args.h, args.mu, args.delta, EXAMPLE_TASK, payoffs=paper_game().payoffs)
    return render_blocks(a, ctx)[0]


def _cmd_prompt(args: argparse.Namespace) -> int:
    blocks = _blocks(args.setting, args.arm, args)
    a = arm(args.setting, args.arm, args.conjecture)
    for name, text in blocks.items():
        variant = a.blocks.get(name, "(generated)")
        print(f"\n===== {name}: {variant} =====\n{text}")
    return 0


def _cmd_prompt_diff(args: argparse.Namespace) -> int:
    a, b = arm(args.setting, args.arm_a), arm(args.setting, args.arm_b)
    ta, tb = _blocks(args.setting, args.arm_a, args), _blocks(args.setting, args.arm_b, args)
    for name in (*BLOCKS, "output"):
        if ta[name] == tb[name]:
            print(f"= {name}: identical ({a.blocks.get(name, 'generated')})")
            continue
        print(f"~ {name}: {a.blocks.get(name, 'generated')} -> {b.blocks.get(name, 'generated')}")
        for line in difflib.unified_diff(ta[name].splitlines(), tb[name].splitlines(), lineterm="", n=0):
            if not line.startswith(("---", "+++", "@@")):
                print("   " + line)
    return 0


def _cmd_arms(_: argparse.Namespace) -> int:
    for setting, registry in (("toy", TOY_ARMS), ("math_qa", MATH_QA_ARMS)):
        print(f"\n{setting}")
        for name, a in registry.items():
            blocks = " ".join(f"{k}={v}" for k, v in a.blocks.items())
            print(f"  {name:36s} {blocks}\n  {'':36s} {a.doc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``smc``."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="smc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run (or resume) an elicitation from a TOML config")
    r.add_argument("config", type=Path)
    r.add_argument("--out", type=Path, help="run directory (default: $SMC_OUTPUT_DIR/runs/<name>)")
    r.add_argument("--dry-run", action="store_true", help="count observations, calls and cost; call nothing")
    r.add_argument("--workers", type=int, default=8, help="parallel requests")
    r.add_argument("--cache", type=Path, help="store every completion here and reuse it")
    r.set_defaults(func=_cmd_run)

    for name, func, help_ in (
        ("prompt", _cmd_prompt, "print an arm's prompt, block by block"),
        ("prompt-diff", _cmd_prompt_diff, "show which blocks two arms differ in"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("setting", choices=["toy", "math_qa"])
        if name == "prompt":
            p.add_argument("arm")
            p.add_argument("--conjecture", action="store_true")
        else:
            p.add_argument("arm_a")
            p.add_argument("arm_b")
        p.add_argument("--h", type=float, default=0.5)
        p.add_argument("--mu", type=float, default=0.5)
        p.add_argument("--delta", type=float, default=0.45)
        p.add_argument("--draw", choices=["easy", "hard"], default="hard")
        p.set_defaults(func=func)

    sub.add_parser("arms", help="list the prompt arms and their blocks").set_defaults(func=_cmd_arms)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
