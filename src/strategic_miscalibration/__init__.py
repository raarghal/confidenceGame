"""Strategic miscalibration: a repeated signaling game for LLM confidence reports.

An *agent* observes a private success probability and reports a confidence signal; a
*user* sees only the signal and chooses to delegate the task or complete it itself.
Repeated play makes the signal a reputation instrument, and the question is whether an
LLM agent misreports to manage it.

Layers, with dependencies pointing strictly down::

    cli -> figures -> analysis -> elicit -> settings -> {llm | prompts} -> theory -> core

- :mod:`~strategic_miscalibration.theory` is the paper's mathematics: exact, and free of
  IO, plotting and model clients.
- :mod:`~strategic_miscalibration.settings` turn a run configuration into observations,
  so the runner in :mod:`~strategic_miscalibration.elicit` is the same for every setting.
- :mod:`~strategic_miscalibration.llm` carries fake and cached backends, so runs are
  exercised offline with no API key; the paper is reproduced from the shipped data.
"""

__version__ = "0.1.0"
