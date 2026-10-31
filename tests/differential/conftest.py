"""The differential suite: Jolt against omi_physics, the reference engine.

``test_scenes.py``, ``test_smoke.py``, ``test_characters.py`` and ``test_p3.py``
compare with omi_physics' recorded answers and run without it; the vehicle
scenarios (``test_vehicles.py``, ``vehicles.py``) run omi_physics live and are
not collected without it.
"""

from __future__ import annotations

from differential.engines import have_omi

collect_ignore = [] if have_omi() else ['test_vehicles.py', 'vehicles.py']
