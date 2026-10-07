# The model in code

## The game as axes

`theory.game.Game` combines independent choices. Each generalization the project anticipates is a
different value on one axis, never a new code path.

| Axis | Type | The paper | Other values |
|---|---|---|---|
| Payoffs | `Payoffs(reward, cost, effort)` | `r = 1, c = 0.1, e = 0.5`, so `ρ* = 1 − (e − c)/r = 0.6` | any `0 < c < e ≤ r` |
| Task and types | `TypeSpace` | abilities `θ_H = 0.8`, `θ_L = 0.2` (each the chance of an easy draw); success `ρ⁺ = 0.85`, `ρ⁻ = 0.15` | graded success `TypeSpace.graded(...)`; more draws |
| Reports | `ReportSpace` | `LOW`/`HIGH` at `ρ⁻`/`ρ⁺` | `ReportSpace.grid(k)`: `k` confidence levels |
| Monitoring | `Monitoring(audit_prob, audit_cost)` | endogenous: an outcome is seen only if delegated | `EXOGENOUS`; an audit with probability `q` |
| Market | `Roster` | the strategic agent alone | `TransparentAgent`: an honest competitor of known ability |
| Horizon | `Game.horizon` | two periods | more (the Bayes engine is horizon-free) |

`paper_game(**overrides)` builds the paper's game with any axis replaced.

**Types and beliefs.** A type is a cell `(η, ability)`: honesty `η` (1 honest, 0 strategic) and ability. The
user's belief is a *law* on cells, the joint law, which is not a product once a non-honest report has been
seen. Theorem 4.1's terminal test is stated on the joint law. This is why the paper's `(h, μ)` marginals
are not a sufficient state.

**Coupled draws.** Each period one latent difficulty `ω ~ U[0, 1]` is drawn, shared by every agent. An
agent of ability `a` gets the draw whose interval of `a`'s cumulative draw distribution contains `ω`. With
one agent this is just `P(draw | a)`. With a competitor it is the draw–audit coupling of the competition
model (the draw–audit coupling), with no special-case code.

## The Bayes engine

`theory.beliefs` is the only place a posterior is computed. Its main functions:

- `weights(game, law, strategy, reports, action, outcomes)` — unnormalized posterior weights after one
  period's public events: every agent's report, the user's action, and the outcomes the monitoring reveals.
- `posterior(...)` — the same, normalized.
- `expected_success(...)` — the paper's `ρ̃`.

`Strategy` is the strategic type's rule `σ(message | ability, draw)`; `Strategy.binary(σ⁺, σ⁻)` is the paper's.

## Where the paper's objects live

| Paper | Code |
|---|---|
| `ρ*`, `Ψ*`, `κ = δ/(1 − δ)`, `δ* = 0.459` | `Payoffs.rho_star`, `terminal.psi_star`, `game.kappa`, `Game.delta_star` |
| Thm 4.1: terminal trust test, trust index `Ψ̂ = P(easy)/P(strategic ∧ hard)` | `terminal.is_trusted`, `terminal.final_success_gap`, `terminal.trust_index` |
| Trust regions `τR`, `τF`, `D` | `regions.region` |
| First-period equilibria: corner, edge (PS, PI1, HI1, HS), boundary (SD3, SD3′, SD6, BI1–BI3) | `solvers.binary_t2.equilibria` |
| Payoff relevance, agent-optimal selection | `Profile.payoff_relevant`, `binary_t2.agent_optimal` |
| Proposition H.5 | `binary_t2.sigma_bar_closed_form` |
| Thm 4.3's standard-user rate at a state | `regions.standard_user_sigma_minus` |
| The Bayes-rational user | `users.OracleUser` |
| Welfare: `W_0`, `W_honest`, naive, sophisticated, informed users | `welfare.two_periods`, `welfare.benchmark`, `welfare.user` |

`solvers.solve(game, h, μ, δ)` dispatches to the first registered solver whose `unsupported(game)` returns
`None`. The paper's solver covers a monopoly, binary reports, the ungraded binary type space, endogenous
monitoring and two periods. Other games raise `NoSolverError`, with each solver's reason for declining.

## How the theory is checked

`tests/theory/` holds the exact checks behind the theorems. Each compares the package against the paper's
own equations, or against `tests/theory/bruteforce.py`: an independent enumerator that shares no code with
the package. The default run uses a quick sample; `--theory-draws 600` runs the size the appendix quotes.

| Paper item | Test |
|---|---|
| Thm 4.1, the odds identities, the blind-trust maximin | `test_final_period.py` |
| Thm 4.2, the ordering of the trust index under truthful reports, Example F.2 | `test_final_period.py` |
| First-period closed forms | `test_equation_bank.py` |
| Proposition I.1, Example I.2 | `test_boundary_families.py` |
| Thms 4.2–4.4, Theorem H.9, the watershed | `test_first_period_set.py` |
| The solver against brute force | `test_solver_vs_bruteforce.py` |
| The solver against recorded reference equilibria, profile for profile | `test_solver_parity.py` |
| Welfare propositions W1–W5 | `test_welfare_propositions.py` |
| Regions and rates on the experiment grid | `test_regions.py` |
