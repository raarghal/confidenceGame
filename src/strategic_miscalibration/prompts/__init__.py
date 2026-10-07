"""What the agent is asked: five-block prompts, the arms built from them, and the answers they expect.

- :mod:`.compose` — the five blocks (``role``, ``setup``, ``state``, ``query``, ``output``), :class:`~.compose.Arm`
  and rendering.
- :mod:`.arms` — every arm the paper's runs used, as derivations that change one block at a time.
- :mod:`.schemas` — the answer each ``query`` variant asks for.
- :mod:`.output_format` — the JSON format block generated from a schema.
"""
