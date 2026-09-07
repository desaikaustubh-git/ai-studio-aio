"""AI Studio H3 Prompt Writer - a ComfyUI custom node.

Turns a plain-language video description into a finished MiniMax H3 prompt by
running the "H3 Prompt Formulas" rule set (the same instructions the Ep29
"Custom ChatGPT" uses) through an LLM, then hands the prompt straight to the
MiniMax H3 video nodes.

Two backends, chosen on the node:
  * chatgpt  - drives your signed-in ChatGPT web session via a bundled copy of
               the browser_llm.py Playwright automation (no API key). Default.
               browser_llm.py + the logged-in Chromium profile are vendored under
               ai_studio_h3_prompt/vendor/ so this folder is relocatable.
  * ollama   - direct HTTP to a local Ollama server (qwen3:32b by default).

Also here:
  * ``AIStudioH3MultiShotRender`` ("H3 Multi-Shot Lengthy Video") - takes one long
    script, has the LLM split it into an ordered shot plan (each shot a full H3
    prompt + a continue-from-previous-frame flag), then renders every shot in order
    and concatenates them into one continuous IMAGE + AUDIO pair. A staging /
    physical-truth pass (always on) forces every shot prompt to pin down a stage
    plan, an explicit eyeline per person, timed held beats, a continuity capsule
    and hard physical rules (solid bodies, real contact, stable anatomy, clean
    surfaces, its-own-restriction legible text, two-figure contact staged close to
    a locked camera) - the H3 failure modes seen in testing, closed at the prompt
    so no review step is needed. An "advanced film-making" pass (on by default)
    then folds a camera-move library (``system_prompts/camera_moves.md`` /
    ``output/sample_prompts.md``) and ``system_prompts/film_extensions.md``
    (cinematography building blocks + named feature-film "house looks") into every
    shot's prompt without weakening that staging; ``film_look`` picks the look.
  * ``AIStudioH3ScriptWriter`` / ``AIStudioH3StoryExpander`` / ``AIStudioH3Director`` /
    ``AIStudioH3ScriptReview`` / ``AIStudioH3VideoReview`` (script_tools.py) - write a
    hooky shot-by-shot script, expand a short story into a longer detailed one (~2.5 min
    by default), break a script down into a Christopher-Nolan-style shot list (blocking,
    props + positions, camera, lighting, score, SFX per shot), have Gemini restructure it
    (hook first, then context, then story) and check it for logic / physics, and have
    Gemini check the finished video's frames against that script.
"""

from .prompt_writer import (
    NODE_CLASS_MAPPINGS as _PW_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _PW_NAMES,
)
from .multi_shot import (
    NODE_CLASS_MAPPINGS as _MS_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MS_NAMES,
)
from .script_tools import (
    NODE_CLASS_MAPPINGS as _ST_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _ST_NAMES,
)

# animation-style render nodes (2D / 3D / Ghibli / Anime) -- each in its own file,
# each backed by an isolated styles/style_<key>.py spec
_STYLE_CLASSES, _STYLE_NAMES = {}, {}
for _m in ("style_render_2d", "style_render_3d", "style_render_ghibli", "style_render_anime",
           "style_render_photoreal"):
    try:
        _mod = __import__(f"{__name__}.{_m}", fromlist=["*"])
        _STYLE_CLASSES.update(_mod.NODE_CLASS_MAPPINGS)
        _STYLE_NAMES.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as _e:  # noqa: BLE001 -- a broken style must not sink the pack
        print(f"[ai_studio_h3] style node {_m} failed to load: {_e}")

NODE_CLASS_MAPPINGS = {**_PW_CLASSES, **_MS_CLASSES, **_ST_CLASSES, **_STYLE_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_PW_NAMES, **_MS_NAMES, **_ST_NAMES, **_STYLE_NAMES}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
