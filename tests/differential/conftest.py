"""The differential suite: Jolt against omi_physics, the reference engine.

omi_physics' side of every comparison is its recorded answer
(``omi_golden.json``, read by :func:`differential.engines.reference`), so the
suite runs without omi_physics. Where omi_physics is installed each recorded
answer is also checked against a live run.
"""
