"""Photoreal — the pipeline's original/default look, made an explicit, selectable
style so it sits alongside 2D / 3D / Ghibli / Anime instead of being an implicit
"no style set" fallback.

Unlike the other four, this style keeps ``h3-realism-people`` in the video LoRA
stack (it is what makes this look photoreal in the first place) and its plate
prefix is byte-identical to the legacy ``PLATE_STYLE`` constant in orch.py /
``_REF_STYLE`` in multi_shot.py. Selecting this node should render the same as
the plain ``AIStudioH3MultiShotRender`` node's default output; it exists so all
five looks can be picked the same way, and so a future edit to "the photoreal
house look" has one file to touch instead of the two scattered constants.

Isolated from the other styles -- edit freely.
"""

from ._base import StyleSpec, TURBO_LORA, REALISM_LORA

STYLE = StyleSpec(
    key="photoreal",
    label="Photoreal (default house look)",

    prompt_style_block=(
        "Photorealistic cinematic horror aesthetic: naturalistic photographic "
        "detail, believable real-world skin and material textures, cold sick "
        "fluorescent green-white and CRT-green practical light sources, deep "
        "black shadows, heavy analog film grain, muted desaturated grade -- the "
        "established house look for this pipeline."
    ),
    prompt_anti=(
        "no cel shading, no flat 2D outlines, no painterly or hand-drawn look, "
        "no CG-stylised or anime proportions; keep skin, materials, lighting and "
        "grain fully photographic and true to the reference plates."
    ),

    plate_backend="gemini",
    plate_style_prefix=(
        "photorealistic reference still, vertical 9:16, cinematic urban-legend "
        "horror, cold sick fluorescent green-white and CRT-green light, deep "
        "black shadows, heavy analog film grain, muted desaturated palette."
    ),
    plate_checkpoint="",
    plate_loras=(),

    # video stack: identical to BRANCHES["turbo"] / BRANCHES["noturbo"] today,
    # including the h3-realism-people LoRA the other four styles drop.
    video_turbo_loras=((TURBO_LORA, 1.0), (REALISM_LORA, 0.9)),
    video_turbo_shift=(6.0, 3.0),
    video_turbo_steps=8,
    video_turbo_sampler="euler",
    video_turbo_sched="simple",
    video_noturbo_loras=((REALISM_LORA, 0.9),),
    video_noturbo_shift=(0.0, 3.0),
    video_noturbo_steps=20,
    video_noturbo_sampler="res_multistep",
    video_noturbo_sched="simple",

    house_look_name="Photoreal (default)",
    house_look_body=(
        "Whole film is grounded photoreal horror. Lighting is practical and "
        "diegetic: sick fluorescent green-white and CRT-green sources, deep "
        "crushed blacks, no CG gloss. Grain and grade stay filmic and "
        "desaturated. Camera holds still, locked-off tension over movement; a "
        "slow push only when the beat earns it. Score: low sustained dread "
        "tones. Sound: room hum, silence, one hard sting on the sting-beat."
    ),
)
