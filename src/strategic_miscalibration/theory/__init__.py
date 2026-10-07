"""The Confidence Game's mathematics: exact, and free of IO, plotting and model clients.

- :mod:`.game` — the game as orthogonal axes (types, reports, monitoring, roster, payoffs).
- :mod:`.beliefs` — the exact joint-law Bayes engine; the only place a posterior is computed.
- :mod:`.terminal` — the final-period trust test, trust index and trust regions (Thm 4.1).
- :mod:`.solvers` — first-period equilibria, dispatched by the games each solver covers.
- :mod:`.users` — the Bayes-rational user.
- :mod:`.welfare` — the user's payoff under the measured and the equilibrium rules.
"""
