"""Shared factory for the four animation-style render nodes.

Each ``style_render_<key>.py`` is a three-line file that hands its StyleSpec to
``make_style_node``. The heavy render path stays in ``multi_shot.py`` (one place
to fix bugs); everything that differs between styles lives in
``styles/style_<key>.py``. Editing one style file cannot affect the others.
"""

from __future__ import annotations

import copy

from .multi_shot import AIStudioH3MultiShotRender
from .styles._base import StyleSpec


def _default(opt: dict, name: str, value) -> None:
    """Override the ComfyUI widget default for ``name`` if it is a (type, cfg) pair."""
    cur = opt.get(name)
    if isinstance(cur, tuple) and len(cur) == 2 and isinstance(cur[1], dict):
        opt[name] = (cur[0], {**cur[1], "default": value})


def make_style_node(spec: StyleSpec):
    class _StyleNode(AIStudioH3MultiShotRender):
        STYLE = spec
        CATEGORY = "AI Studio/H3/Style"
        _rl_name, _rl_strength = spec.realism_lora     # ("", 0.0) for every style but photoreal
        DESCRIPTION = (f"H3 multi-shot render locked to the {spec.label} look: "
                       f"{'h3-realism-people on' if _rl_name else 'h3-realism-people off'}, "
                       f"the style block folded into every shot prompt, reference "
                       f"stills restyled to match.")

        @classmethod
        def INPUT_TYPES(cls):
            t = copy.deepcopy(AIStudioH3MultiShotRender.INPUT_TYPES())
            opt = t.setdefault("optional", {})
            rl_name, rl_strength = spec.realism_lora
            _default(opt, "steps", spec.video_turbo_steps)
            _default(opt, "sampler_name", spec.video_turbo_sampler)
            _default(opt, "scheduler", spec.video_turbo_sched)
            _default(opt, "speed_lora", spec.turbo_lora)
            _default(opt, "speed_lora_strength", 1.0)
            _default(opt, "quality_lora", rl_name)       # "" for every style but photoreal
            _default(opt, "quality_lora_strength", rl_strength)
            _default(opt, "sigma_shift", spec.video_turbo_shift[0])
            _default(opt, "sigma_shift_audio", spec.video_turbo_shift[1])
            _default(opt, "film_look", "none")
            _default(opt, "advanced_prompts", False)
            opt["style_intensity"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Reserved knob for how hard to lean on the style block "
                           "(1.0 = full). Currently informational."})
            return t

        def run(self, **kw):
            rl_name, rl_strength = spec.realism_lora
            kw["quality_lora"] = rl_name                              # "" unless photoreal
            kw["quality_lora_strength"] = rl_strength
            if not str(kw.get("speed_lora") or "").strip():
                kw["speed_lora"] = spec.turbo_lora
            if not float(kw.get("sigma_shift") or 0.0):
                kw["sigma_shift"] = spec.video_turbo_shift[0]
            kw.pop("style_intensity", None)
            kw["style_prompt_block"] = spec.style_prompt("")
            kw["ref_style_override"] = spec.plate_style_prefix
            return super().run(**kw)

    _StyleNode.__name__ = f"AIStudioH3Render{spec.key.capitalize()}"
    _StyleNode.__qualname__ = _StyleNode.__name__
    return _StyleNode
