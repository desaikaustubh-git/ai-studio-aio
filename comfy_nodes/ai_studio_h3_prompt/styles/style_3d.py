"""3D Pixar-style CG feature-animation look.

No Pixar-specific checkpoint ships on this box, so plate generation defaults to
the "gemini" web image generator (it renders this look well with zero new files).
To move plates local: drop a stylised-3D SDXL checkpoint into
models/checkpoints, set plate_checkpoint below and flip plate_backend to
"comfy_sd". Edit this file freely -- it is isolated from the other styles.
"""

from ._base import StyleSpec

STYLE = StyleSpec(
    key="3d",
    label="3D Pixar-style CG",

    prompt_style_block=(
        "Rendered as a stylised 3D computer-animated feature in the modern Pixar "
        "house style: rounded appealing character design with gently exaggerated "
        "proportions and large expressive eyes, soft global illumination with warm "
        "bounce light, believable subsurface scattering on skin, clean matte and "
        "lightly worn surfaces, tasteful ambient occlusion, shallow cinematic "
        "focus falloff, physically plausible cloth and hair simulation. Polished "
        "CG-animation render, not live action."
    ),
    prompt_anti=(
        "no photoreal live-action footage, no documentary realism, no harsh video "
        "noise or sensor grain, no flat 2D linework or cel shading, no uncanny "
        "hyper-detailed pores; keep character model proportions and shader look "
        "identical shot to shot."
    ),

    plate_backend="gemini",
    plate_style_prefix=(
        "3D Pixar-style computer-animated feature still, stylised appealing "
        "character design, soft global illumination, subsurface-scatter skin, "
        "clean matte surfaces, shallow depth cues, vertical 9:16 composition,"
    ),
    # --- comfy_sd fields kept ready for when a local 3D checkpoint is added ------
    plate_checkpoint="",           # e.g. a Pixar-style SDXL checkpoint
    plate_loras=(),
    plate_sampler="dpmpp_2m",
    plate_scheduler="karras",
    plate_steps=32,
    plate_cfg=6.5,

    house_look_name="Pixar-style 3D Feature",
    house_look_body=(
        "Whole film is stylised 3D CG animation. One warm motivated key with soft "
        "global-illumination fill; gentle ambient occlusion in the contact "
        "shadows; hero lighting on faces. Palette rich but controlled, a clear "
        "warm/cool separation. Camera behaves like a real cinema rig on a virtual "
        "dolly and crane -- smooth, motivated, one move per shot. Materials read "
        "as slightly stylised, never grimy-photoreal. Score: full orchestral with "
        "a memorable theme; clean designed foley."
    ),
)
