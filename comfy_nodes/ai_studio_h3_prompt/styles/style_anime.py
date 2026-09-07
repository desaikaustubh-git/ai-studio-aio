"""Modern Japanese TV / feature anime look.

Plates render locally on this box's anime SDXL checkpoint. Isolated from the
other styles -- edit freely.
"""

from ._base import StyleSpec

STYLE = StyleSpec(
    key="anime",
    label="Modern Anime",

    prompt_style_block=(
        "Rendered as modern Japanese anime: crisp thin clean linework, flat cel "
        "shading with hard-edged shadow shapes in two tones, bright high-saturation "
        "key colours, detailed painted background art with soft light bloom, "
        "anime-style rim light and sharp specular highlights in the eyes and hair, "
        "expressive posing. Speedlines and impact frames only when the motion "
        "genuinely calls for them."
    ),
    prompt_anti=(
        "no photographic realism, no film grain or lens bokeh, no 3D-rendered "
        "look, no muddy desaturated grade, no western-comic ink texture; keep "
        "character colour scripts and line thickness consistent shot to shot."
    ),

    plate_backend="comfy_sd",
    plate_style_prefix=(
        "modern anime key visual, crisp clean linework, flat two-tone cel "
        "shading, saturated key colours, detailed painted background, anime rim "
        "light and eye highlights, vertical 9:16 composition,"
    ),
    plate_checkpoint="anyloraCheckpoint_bakedvaeBlessedFp16.safetensors",
    plate_loras=(),
    plate_sampler="euler_ancestral",
    plate_scheduler="normal",
    plate_steps=28,
    plate_cfg=7.0,

    house_look_name="Modern Anime",
    house_look_body=(
        "Whole film is modern anime. Lighting is graphic: hard two-tone cel "
        "shadows, blown highlights, coloured rim light separating the figure from "
        "a detailed painted background. Palette bright and saturated with a clear "
        "key colour per scene. Camera uses anime grammar -- locked held frames, "
        "quick whip-pans, a slow push for tension, an occasional dramatic "
        "dolly-zoom on a reveal. Score: synth-and-strings or a driving band cue. "
        "Sound: crisp designed effects, deliberate silences before an impact."
    ),
)
