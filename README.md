# Strategic Miscalibration

Code for *The Confidence Game: Strategic Miscalibration in Human-AI Delegation*. It does two things:

- **Reproduces the paper.** One script per results section regenerates that section's figure and every
  number in its text from the shipped data.
- **Runs the experiments.** Elicits reporting strategies from an LLM in the paper's two settings: a toy
  setting, where the agent is told its success probability, and a math Q&A setting, where it answers real
  questions.

An **agent** of unknown honesty and ability reports its confidence. A **user** sees only the report and
decides whether to delegate the task, paying a fee, or to do it herself. Play repeats, so the report also
manages the agent's reputation. This work studies whether an LLM agent misreports to manage it, and what
that costs the user.

## Requirements

[uv](https://docs.astral.sh/uv/) and Python 3.12 or newer; `uv sync` installs Python itself if you don't
have it. Run every command below from inside the repository.

## Reproducing the paper

Offline, no API key, a few seconds each:

```bash
uv sync
uv run python scripts/section5.py     # Section 5, LLM Experiments: Figure 2 and its numbers -> out/section5/
uv run python scripts/section6.py     # Section 6, Welfare Impact:  Figure 3 and its numbers -> out/section6/
```

Each prints its numbers beside the values the paper prints.

The appendix is reproducible too, less directly: [`docs/reproducing.md`](docs/reproducing.md) lists the
analysis function and the data behind each appendix table and figure.

## Running your own experiments

```bash
uv sync --extra llm
cp .env.example .env                  # then add your API key
uv run python scripts/run_experiment.py configs/example.toml --dry-run    # calls and cost; nothing is sent
uv run python scripts/run_experiment.py configs/example.toml              # 8 calls, about a cent
```

Calls are sent 8 at a time; `--workers N` changes that (lower it if the provider rate-limits you).
Rows are written to `out/runs/<name>/`, and rerunning the same command resumes an interrupted run.
`configs/example.toml` is a small run to copy and edit: the model, the prompt arms, the grid of states and
the sample count; `configs/example_math_qa.toml` is its math Q&A counterpart. `configs/toy/` and `configs/math_qa/` hold the configurations behind the paper's runs.
To see what a model is sent:

```bash
uv run smc arms                                   # the prompt arms
uv run smc prompt toy minimal_clarified_ledger    # one arm's prompt, block by block
```

The `.env` is read from the working directory or the nearest directory above it, and a variable already set
in the shell takes precedence. Models are called through LiteLLM, so any provider it supports works with
that provider's key.

## Layout

```
scripts/      section5.py, section6.py (the paper's main body), run_experiment.py (your own runs)
configs/      example.toml, and one TOML per run behind the paper (toy/, math_qa/)
data/runs/    those runs: rows.csv.gz + config.toml + manifest.json
src/strategic_miscalibration/
  theory/     the game, the exact Bayes engine, solvers, the oracle user, welfare
  prompts/    five-block prompts, the paper's arms, answer schemas
  llm/        model capabilities, live / fake / cached backends, parsing, a parallel client
  settings/   the toy and math Q&A settings: prompt contexts and observations
  elicit/     TOML run configurations and a resumable runner
  analysis/   behavior, theory_fit, welfare, transfer: one module per question asked
  figures/    the paper's figures
  cli/        `smc`
tests/        theory (exact checks against an independent brute-force enumerator), integration (offline runs)
docs/         model.md, reproducing.md, extending.md, data.md
```

## Documentation

- [`docs/reproducing.md`](docs/reproducing.md) — the main body, the appendix, and rerunning the experiments from scratch.
- [`docs/data.md`](docs/data.md) — the shipped runs, their columns and their provenance.
- [`docs/model.md`](docs/model.md) — the game, and where each theorem's objects live in the code.
- [`docs/extending.md`](docs/extending.md) — recipes: a new arm, setting, model, solver or user; graded success, auditing, a competitor.

## Tests

```bash
uv sync --all-groups
uv run pytest
```

## Citation
If you find this repository useful, please consider citing:

```
@article{arghal2026ConfidenceGame,
  title={The Confidence Game: Strategic Miscalibration in Human-AI Delegation},
  author={Arghal, Raghu and Sarkar, Saswati and Bidokhti, Shirin Saeedi},
  year={2026}
}
```

## Licensing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This work is licensed under the MIT License (MIT)

Copyright (c) 2026 Raghu Arghal
