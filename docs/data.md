# Data

`data/runs/<run>/` holds the 14 runs behind the paper, about 11 MB in total.

| File | Contents |
|---|---|
| `rows.csv.gz` | one row per elicitation, reasoning traces included |
| `config.toml` | the run's configuration (a copy of `configs/<setting>/<run>.toml`) |
| `manifest.json` | row and valid counts, schema version, and the SHA-256 of the raw run file it was packed from |

Load a run with `analysis.io.load_run("<run>")`. It returns the rows, configuration and manifest, with
booleans typed. Set `SMC_DATA_DIR` to read runs from somewhere else, such as your own reruns.

## Columns

**Toy setting.** One row per round-1 elicitation at an asserted state.

| Column | Meaning |
|---|---|
| `arm`, `conjecture` | prompt arm; whether the agent first stated its conjecture about the user |
| `h`, `mu`, `delta` | the state: belief in honesty, belief in high ability, weight on this round |
| `agent_ability`, `draw`, `rho_t` | the agent's own ability; `EASY` or `HARD`; its success probability |
| `signal`, `report_high` | the report; whether it is HIGH |
| `prob_high` | strategy arms: the stated probability of reporting HIGH |
| `conj_delegate_high`, `conj_delegate_low` | belief-elicited arms: the stated chance the user delegates after each report |
| `payoff_{this,next}_round_if_{high,low}` | ledger arm: the four stated round payoffs, recorded raw |
| `reasoning`, `is_valid` | the reasoning trace; whether the answer parsed to a valid report |

**Math Q&A setting.** One row per question, asked plainly and in the game.

| Column | Meaning |
|---|---|
| `arm`, `h`, `mu`, `delta`, `sample_idx`, `difficulty` | design: arm, state, SuperGPQA row (mathematics subset), difficulty |
| `rho`, `baseline_solution`, `baseline_correct` | plain prompt: confidence (the agent's own `ρ`), answer, correct |
| `s`, `solution`, `correct` | game prompt: report, answer, correct |
| `gold` | the answer key's letter |
| `ledger_candidates` | ledger arm: priced candidate reports, as JSON |
| `agent_raw`, `agent_rung` | the raw game response and how it was parsed |

`correct` and `baseline_correct` resolve value answers against the option texts. The game framing often
produces answers like "12" instead of "C". Grading uses `settings.math_qa.grade`, the same function new
runs use. The first-letter grades recorded when the runs were made are kept as `*_first_letter`.

## Provenance

Every run used `together_ai/openai/gpt-oss-120b` at temperature 0.7, in text mode with the reasoning
channel. Runs from this package write the same schema (`elicit.runner.SCHEMA_VERSION`), plus each call's
raw response, parse rung, failure kind and cost.
