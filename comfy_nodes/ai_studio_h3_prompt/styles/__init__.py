"""Animation style registry -- shared by orch.py and the ComfyUI node package.

Each style lives in its own ``style_<key>.py`` module and exposes a single
``STYLE = StyleSpec(...)``.  Nothing here mutates across styles, so adding or
editing one style cannot affect the others.

    from styles import get
    spec = get("anime")          # -> StyleSpec | None
    branch = spec.apply_branch(BRANCHES["turbo"])
"""

from ._base import StyleSpec, sd_plate_graph, TURBO_LORA, REALISM_LORA
from . import style_2d, style_3d, style_ghibli, style_anime, style_photoreal

_MODULES = (style_2d, style_3d, style_ghibli, style_anime, style_photoreal)
STYLES = {m.STYLE.key: m.STYLE for m in _MODULES}
ALL_KEYS = tuple(STYLES.keys())


def get(key):
    """StyleSpec for ``key`` (case-insensitive), or None for a falsy/unknown key."""
    if not key:
        return None
    return STYLES.get(str(key).strip().lower())


__all__ = ["StyleSpec", "sd_plate_graph", "get", "STYLES", "ALL_KEYS",
           "TURBO_LORA", "REALISM_LORA"]
