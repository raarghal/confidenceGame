"""Talking to models: capabilities, backends (live, fake, cached), parsing, and a parallel client.

Nothing here knows about the game or the prompts. It takes a prompt and a schema and returns a parsed
answer, or a recorded failure.
"""
