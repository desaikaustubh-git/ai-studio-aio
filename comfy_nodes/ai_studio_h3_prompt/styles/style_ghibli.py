"""Studio Ghibli hand-painted 2D look.

No Ghibli-specific checkpoint ships on this box, so plate generation defaults to
the "gemini" web image generator. To move plates local: add a Ghibli SDXL
checkpoint or LoRA under models/, set the fields below and flip plate_backend to
"comfy_sd". Isolated from the other styles.
"""

from ._base import StyleSpec

STYLE = StyleSpec(
    key="ghibli",
    label="Studio Ghibli",

    prompt_style_block=(
        "Rendered as hand-painted 2D animation in the Studio Ghibli style: soft "
        "naturalistic watercolour and gouache backgrounds with visible brushwork "
        "and layered foliage, gentle even linework, restrained character shading "
        "with soft flat shadows, a muted earthy-pastel palette, big flat sky "
        "gradients with towering cumulus clouds, warm nostalgic daylight, quiet "
        "unhurried movement. Everything looks painted by hand."
    ),
    prompt_anti=(
        "no photographic realism, no lens grain or bokeh, no 3D-rendered surfaces, "
        "no hard digital gradients or neon saturation, no thick black comic "
        "outlines; keep the painterly background treatment and palette consistent "
        "shot to shot."
    ),

    plate_backend="gemini",
    plate_style_prefix=(
        "Studio Ghibli hand-painted animation still, soft watercolour background, "
        "gentle linework, muted earthy-pastel palette, big painted sky, warm "
        "nostalgic light, vertical 9:16 composition,"
    ),
    plate_checkpoint="",
    plate_loras=(),
    plate_sampler="euler",
    plate_scheduler="normal",
    plate_steps=30,
    plate_cfg=6.0,

    house_look_name="Ghibli Hand-Painted",
    house_look_body=(
        "Whole film is hand-painted 2D animation. Light is soft, natural and "
        "motivated by sky or window; long calm holds on landscape; weather and air "
        "carry emotion. Palette muted and earthy with clean pastel skies. Camera "
        "is mostly locked with slow, patient pans and the occasional gentle push; "
        "no snappy moves. Score: solo piano or small orchestra, spare and "
        "melodic. Sound: rich natural ambience -- wind, grass, water, distant "
        "birds -- and pockets of near-silence."
    ),
)
