"""Run your own elicitation: send a configuration's prompts to a model and save its reports.

    uv sync --extra llm
    cp .env.example .env                      # then add your API key
    uv run python scripts/run_experiment.py configs/example.toml --dry-run    # calls and cost; nothing is sent
    uv run python scripts/run_experiment.py configs/example.toml

Rows go to out/runs/<name>/rows.csv.gz (--out to change). Rerunning the same command resumes an interrupted
run. A configuration names the setting (toy or math_qa), the model, the prompt arms and the grid of states;
configs/toy/ and configs/math_qa/ hold the ones behind the paper.

Calls are sent 8 at a time; --workers N changes that. --cache DIR stores every completion and reuses it.
This is `smc run`.
"""

from __future__ import annotations

import sys

from strategic_miscalibration.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
