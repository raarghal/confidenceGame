# Reproducing the paper

Everything here runs offline from the runs shipped in `data/runs/` (see [`data.md`](data.md)).

## The main body

```bash
uv run python scripts/section5.py     # Section 5: Figure 2 and every number in the text -> out/section5/
uv run python scripts/section6.py     # Section 6: Figure 3 and every number in the text -> out/section6/
```

Each script prints its numbers beside the values the paper prints. Figure 1, the timing diagram, is drawn
by hand.

## The appendix

Every appendix number comes from a function in
`src/strategic_miscalibration/analysis/`, and each function reads the runs it needs
and returns a table or a dictionary. Call the one behind the number you want:

```python
# uv run python
from strategic_miscalibration.analysis import behavior, theory_fit, transfer, welfare

theory_fit.region_table()  # Table 7: σ⁻ by region and δ, measured and in equilibrium
behavior.protocol_slopes("cued")  # Table 6, one protocol: slopes in h, μ and δ with intervals
welfare.protocol_lever()  # Appendix M: informativeness and welfare cost by protocol
```

| Paper | Function | Runs |
|---|---|---|
| **Appendix M, protocol sensitivity** | | |
| Tables 5, 6: slopes by protocol | `behavior.protocol_slopes(protocol)` | the five protocol runs |
| difference in a slope between two protocols | `behavior.protocol_did(ref, arm, channel)` | the five protocol runs |
| effect of one prompt clause | `behavior.clause_effect(ref, arm)` (rungs: `behavior.CLAUSE_RUNS`) | the four clause-gate runs |
| states that over-report | `behavior.over_reporting_states(protocol)` | the five protocol runs |
| the misreading screen | `behavior.screened(rows)` | `ledger_full` |
| informativeness and welfare cost by protocol | `welfare.protocol_lever()` | the five protocol runs |
| **Appendix N, agreement with the theory** | | |
| Table 7: σ⁻ by region and δ | `theory_fit.region_table()` | `ledger_full` |
| response to δ within each region | `theory_fit.myopia_response()` | `ledger_full` |
| Table 8: match to the equilibrium by protocol | `theory_fit.protocol_match()` (about two minutes) | the five protocol runs |
| ordering of the equilibrium families | `theory_fit.family_ordering()` | `ledger_full` |
| under-reporting | `theory_fit.under_reporting()`, `theory_fit.low_reports_on_easy_draws()` | `ledger_full` |
| the agent's beliefs about the user | `theory_fit.belief_mediation()` | `ledger_full` |
| Table 9: substituting stated beliefs into the theory | `theory_fit.substitution()`, `theory_fit.model_jumps()` | `ledger_full` |
| stated continuation values | `theory_fit.stated_continuation()` | `ledger_full` |
| **Appendix N.1, checks** | | |
| reports maximize the agent's stated payoffs | `theory_fit.coherence()` | `ledger_full` |
| payoff forgone | `theory_fit.forgone_payoff()` | `g3_clarified_full` |
| accuracy of stated conjectures | `theory_fit.conjecture_accuracy()` | `g3_clarified_full` |
| **Appendix O, real tasks** | | |
| Table 10 and levels | `transfer.levels()`, `transfer.calibration()` | the four `mathqa_*` runs |
| answer format | `transfer.answer_format()` | the four `mathqa_*` runs |
| the binarized reporting rule | `transfer.binarized_rule()` | `mathqa_map_supergpqa` and the toy runs |
| prompt variants | `transfer.prompt_variants()` | the three `mathqa_calib_*` runs |
| **Appendix P, welfare** | | |
| per-state welfare, losses and their split | `welfare.state_table()`, `welfare.summary()` | `ledger_full` |
| capability sweep and break-even shares | `welfare.capability_sweep()`, `welfare.capability_readings()`, `welfare.nine_state_readings()` | `ledger_full` |
| states where exploitation is strict | `welfare.w4_strict_count()` | none (theory) |
| the wrong-population example | `welfare.wrong_population_witness()` | none (theory) |
| the low-ability agent | `welfare.type_independence()` | `wL_gate`, `ledger_full` |

The five protocol runs are `e1_core` (semantics), `g3b_clarified_noconj` (report-only), `g3_clarified_full`
(belief-elicited), `c1_cued_full` (cued) and `ledger_full` (ledger); `behavior.PROTOCOLS` holds the mapping.
The four clause-gate runs are `e0_semantics_check`, `g1b_clarified_noconj`, `l0_link_power` and
`g1_clarified_gate`.

Intervals are bootstrap intervals with fixed seeds, so they are the same on every run.

The two appendix figures:

```python
from pathlib import Path
from strategic_miscalibration.figures import experiments, welfare

experiments.glyph(Path("out/appendix/figure_4.pdf"))  # σ⁺, σ⁻ and the equilibrium at each belief state
welfare.welfare_full(Path("out/appendix/figure_5.pdf"))  # welfare loss for both users at every δ
```

The exact theory audits the appendix quotes are the tests in `tests/theory`:

```bash
uv run pytest tests/theory --theory-draws 600
```

## From scratch (regenerating the data)

Each run behind the paper has a configuration in `configs/`. Running one needs the `llm` extra and a
Together API key. Costs are estimates from the measured average cost per call of the paper's runs
(`smc run … --dry-run` prints them).

| Run | Config | Observations | Calls | Est. cost |
|---|---|---|---|---|
| ledger (main toy run) | `configs/toy/ledger_full.toml` | 4,000 | 4,000 | $5.20 |
| cued | `configs/toy/c1_cued_full.toml` | 4,000 | 4,000 | $5.20 |
| report-only | `configs/toy/g3b_clarified_noconj.toml` | 4,000 | 4,000 | $5.20 |
| belief-elicited | `configs/toy/g3_clarified_full.toml` | 5,000 | 5,000 | $6.50 |
| semantics | `configs/toy/e1_core.toml` | 5,000 | 5,000 | $6.50 |
| low-ability gate | `configs/toy/wL_gate.toml` | 800 | 800 | $1.04 |
| clause gate: semantics | `configs/toy/e0_semantics_check.toml` | 800 | 800 | $1.04 |
| clause gate: report-only | `configs/toy/g1b_clarified_noconj.toml` | 800 | 800 | $1.04 |
| clause gate: link | `configs/toy/l0_link_power.toml` | 800 | 800 | $1.04 |
| clause gate: belief-elicited | `configs/toy/g1_clarified_gate.toml` | 1,000 | 1,000 | $1.30 |
| math Q&A map | `configs/math_qa/mathqa_map_supergpqa.toml` | 800 | 1,600 | $2.08 |
| math Q&A: prompt length | `configs/math_qa/mathqa_calib_length.toml` | 720 | 1,440 | $1.87 |
| math Q&A: decomposition | `configs/math_qa/mathqa_calib_decomp.toml` | 720 | 1,440 | $1.87 |
| math Q&A: paid on success | `configs/math_qa/mathqa_calib_hazard.toml` | 480 | 960 | $1.25 |

```bash
uv sync --extra llm
cp .env.example .env                  # then add your API key
uv run python scripts/run_experiment.py configs/toy/ledger_full.toml --workers 8 --cache out/cache
SMC_DATA_DIR=out uv run python scripts/section5.py     # analyse your runs instead of the shipped ones
```

The `.env` is read from the working directory or the nearest directory above it. To check a setup before
paying for a full run, run `configs/example.toml`: 8 calls, about a cent.
