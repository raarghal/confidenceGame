# Extending the package

Each recipe below is an addition: a new value, file or registry entry. None changes existing code.

## Graded success (ability changes success on easy draws)

```python
from strategic_miscalibration.theory.game import Game, TypeSpace

game = Game(TypeSpace.graded(theta_H=0.8, theta_L=0.2, rho_plus_H=0.95, rho_plus_L=0.55, rho_minus=0.15))
```

The Bayes engine, the oracle user, welfare and the prompts handle it as is: the toy `setup` block adds the
graded-success sentence by itself. A run uses it by adding `[game] graded = {rho_plus_H = 0.95, rho_plus_L = 0.55}`
to its TOML. The paper's solver declines graded games (`NoSolverError`), so equilibria need a new solver
(see below).

## Quantized confidence reports

```python
from strategic_miscalibration.theory.game import ReportSpace, paper_game

game = paper_game(reports=ReportSpace.grid(5))  # 0, 0.25, 0.5, 0.75, 1; the honest type reports the nearest
```

In the math Q&A setting, set `confidence_mode = "tercile"` (or `binary`, `quartile`) in the run's TOML.
The ledger schema then enumerates one priced pair per admissible report (`prompts.schemas.math_qa_ledger_schema`).

## Auditing (the user can see the outcome without delegating)

```python
from strategic_miscalibration.theory.game import Monitoring, paper_game

game = paper_game(monitoring=Monitoring(audit_prob=0.5, audit_cost=0.02))
```

`audit_prob = 0` is the paper's endogenous monitoring, and `1` is exogenous. Prompts state the audit; in
TOML, `[game] audit_prob = 0.5`.

## An honest competitor

```python
from strategic_miscalibration.theory.game import Roster, TransparentAgent, paper_game

game = paper_game(roster=Roster((TransparentAgent("T", "L"),)))
```

The competitor's report is evidence about the strategic agent's draw through the shared difficulty. A
user who conditions on both reports does this automatically (`users.OracleUser` chooses among self and
every agent). In TOML, `[game] competitor = "L"`.

## A solver for a new game

Write `theory/solvers/<name>.py` with a class implementing `unsupported(game)` and
`equilibria(game, h, mu, delta)`, and register it with `SOLVERS.add("<name>", MySolver())`. Import it in
`theory/solvers/__init__.py`. `solve(...)` dispatches to it for the games it accepts. Compute continuation
values with `beliefs.weights` and `terminal.final_success_gap`, never with your own posterior.

What the paper's solver (`binary_t2`) assumes, and so what a solver for another game has to replace:

- **Two messages and two draws.** A strategy is `(σ⁺, σ⁻)` and a user rule is `(d⁺, d⁻)`; `Profile` hard-codes
  those four fields, and the outcome events are the fixed tuple `("rej", "succ", "fail")`.
- **Two periods.** Continuation values are final-period values from `terminal.final_success_gap`, which covers a
  monopoly with a binary report space only.
- **One unknown at a time.** Every mixed family has one mixing probability and every condition is affine in it, so
  roots are exact.
- **A proved list of families.** It checks corner, edge and boundary profiles, which the paper proves exhaustive
  at generic parameters. Another game has no such proof, so its solver must search: for each pattern of trusted
  final-period beliefs, enumerate the supports each player mixes over, solve the linear indifference systems in
  `Fraction`s, and keep a candidate only if it is consistent with the pattern and both players best-respond.

Before the first new solver: give `Profile` general fields (`d` by message, `σ` as a `Strategy`, values keyed by
event and report) with the binary accessors kept as properties, and build the events from
`beliefs.outcome_events`. Each new solver also needs its own independent brute-force check
(`tests/theory/bruteforce.py` covers the paper's game only) and a parity test against `binary_t2` on the paper's
game (`tests/fixtures/theory/equilibria_joint.json`).

By axis, cheapest first: **auditing** (more events after a rejection; the trust test is unchanged), **graded
success** (`σ` varies by ability, so 4 coordinates; the trust test already holds), **quantized reports** (the full
support search, and a proof that pooling on the top message is still the final-period best response), **a
competitor** (three user actions and a new final-period solver), **a longer horizon** (recursion over the beliefs
reached, with a selection rule at every node; the theory is not written).

## A prompt arm

Every prompt is five blocks: `role`, `setup`, `state`, `query`, `output`. An arm picks a text variant for
each of the first four, and `output` follows from the query's schema. To vary one clause:

1. Copy the block variant you are changing to `prompts/templates/blocks/<block>/<new_variant>.j2` and edit the text.
2. In `prompts/arms.py`, derive the arm from the one it modifies: `NEW = _toy(CUED.with_blocks("my_arm", query="my_variant", doc="what it isolates"))`.
3. If the new query asks for different fields, add its schema to `prompts/schemas.py:QUERY_SCHEMAS`.

`uv run smc prompt-diff toy minimal_clarified_cued my_arm` then shows the difference block by block.
Game facts (monitoring, a competitor, graded success) are not prompt variants; they come from the game.

## A task setting (a new domain)

Write `settings/<name>.py` with a `prompt_context(...)` and an `observations(config, ...)` returning
`settings.base.Observation`s, each with its design columns, its model calls and a `finish(answers) → row`.
Add its prompt blocks and register its arms in `prompts/arms.py`. Route it in `elicit/runner.observations`.

## A different user

Any object with `decide(law, reports) -> Decision` and `update(law, reports, action, outcomes) -> law` is a
user (`theory.users.User`). The oracle is one; an LLM user or a frozen user is another.

## A model

Add a row to `llm/capabilities.py:CAPABILITIES`: whether to ask for plain text, whether reasoning arrives
on its own channel, the reasoning-effort brake, a token budget, and the measured cost per call. Measure
before you trust a parameter: providers can accept one and then ignore it.
