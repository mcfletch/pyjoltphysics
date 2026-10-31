"""The differential suite: Jolt against omi_physics, the reference engine.

``test_scenes.py`` compares with omi_physics' recorded answers and runs without
it; the other modules run omi_physics live and are not collected without it.
"""

from __future__ import annotations

from differential.engines import have_omi

collect_ignore = [] if have_omi() else ['test_characters.py', 'test_p3.py', 'test_smoke.py']
