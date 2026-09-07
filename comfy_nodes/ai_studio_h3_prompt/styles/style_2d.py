"""2D hand-drawn feature-animation look (Disney-Renaissance / traditional cel).

Edit this file freely -- it is imported only by the "2d" style node and the
orch.py style registry, and shares nothing mutable with the other styles.
"""

from ._base import StyleSpec

STYLE = StyleSpec(
    key="2d",
    label="2D Hand-Drawn (Disney Renaissance)",

    prompt_style_block=(
        "Rendered entirely as flat 2D traditional hand-drawn cel animation in the "
        "Disney-Renaissance feature style: clean confident ink outlines of varying "
        "weight, characters built from simple appealing shapes, smooth full "
        "animation with a little squash-and-stretch, hand-painted gouache "
        "background art with soft aerial perspective, warm storybook colour "
        "palette, flat cel shadows in two or three tones, gentle painterly "
        "highlights. No photographic realism anywhere in the image."
    ),
    prompt_anti=(
        "no camera lens artefacts, no photographic film grain, no depth-of-field "
        "blur or bokeh, no motion blur beyond hand-drawn smears, no 3D-rendered "
        "surfaces or ray-traced reflections, no realistic skin pores or subsurface "
        "detail; keep line weight and paint texture consistent shot to shot."
    ),

    plate_backend="comfy_sd",
    plate_style_prefix=(
        "2D hand-drawn cel animation reference art, Disney-Renaissance feature "
        "style, clean ink linework, hand-painted gouache background, flat cel "
        "shading, warm storybook palette, vertical 9:16 composition,"
    ),
    plate_checkpoint="animaPencilXL_v500.safetensors",
    plate_loras=(("batman_adventures_offset.safetensors", 0.55),),
    plate_sampler="dpmpp_2m",
    plate_scheduler="karras",
    plate_steps=30,
    plate_cfg=6.0,

    house_look_name="Hand-Drawn 2D Feature",
    house_look_body=(
        "Whole film is flat 2D hand-drawn cel animation. Lighting is painted, not "
        "simulated: a single warm key implied by cel-shadow shapes, backgrounds "
        "carrying the mood in gouache washes. Palette warm and saturated with a "
        "storybook grade. Staging reads in clean silhouette against painted depth; "
        "camera moves are simple truck / pan / a gentle push on a multiplane feel, "
        "never a real dolly. Score: lush orchestral, a clear melodic theme. Sound: "
        "clean foley, no room-tone hiss."
    ),
)
