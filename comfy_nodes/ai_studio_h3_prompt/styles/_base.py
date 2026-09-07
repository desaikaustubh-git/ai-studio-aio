"""StyleSpec -- the whole contract between an animation style and the two render
pipelines (orch.py batch + the AI Studio ComfyUI node package).

One style == one module that defines a single ``STYLE = StyleSpec(...)``. Nothing
here imports orch.py or ComfyUI, so this file is copied verbatim into the node
package's ``styles/`` folder (canonical copy lives here, hand-sync the four
``style_*.py`` + this file -- same rule as vendored browser_llm.py).

Two things a style changes:
  * the H3 VIDEO stack   -- drop ``h3-realism-people`` (it hard-biases faces to a
    photoreal distribution and fights any stylisation) and swap the LoRA / sigma
    shift / step budget.  H3 has no negative conditioning (CFG 1.0), so the actual
    "look" is carried by ``prompt_style_block`` woven into every shot prompt.
  * the REFERENCE PLATES -- ``plate_style_prefix`` replaces the photoreal
    ``PLATE_STYLE`` / ``_REF_STYLE`` prefix; ``plate_backend`` picks the generator
    ("gemini" web image gen, or "comfy_sd" = a local Stable Diffusion checkpoint
    on this box via ``sd_plate_graph``).
"""

from __future__ import annotations

import dataclasses

TURBO_LORA   = "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"
REALISM_LORA = "h3-realism-people-t2v-i2v-r2v.safetensors"


@dataclasses.dataclass(frozen=True)
class StyleSpec:
    key: str                       # "2d" | "3d" | "ghibli" | "anime"
    label: str

    # --- prompt language folded into every shot (replaces the photoreal prefix) --
    prompt_style_block: str
    prompt_anti: str = ""          # style-specific "never do" clause

    # --- reference-plate generation ------------------------------------------------
    plate_backend: str = "gemini"           # "gemini" | "comfy_sd"
    plate_style_prefix: str = ""            # replaces PLATE_STYLE / _REF_STYLE
    plate_checkpoint: str = ""              # comfy_sd only, from models/checkpoints
    plate_loras: tuple = ()                 # comfy_sd only: ((lora_name, strength), ...)
    plate_sampler: str = "dpmpp_2m"
    plate_scheduler: str = "karras"
    plate_steps: int = 30
    plate_cfg: float = 6.0
    plate_negative: str = ("photograph, photorealistic, realistic skin, 3d render, "
                           "lowres, bad anatomy, extra limbs, watermark, signature, "
                           "text, caption, jpeg artifacts")

    # --- H3 VIDEO stack overrides (merged onto a BRANCHES entry) ------------------
    video_turbo_loras: tuple = ((TURBO_LORA, 1.0),)   # NOTE: no realism LoRA
    video_turbo_shift: tuple = (6.0, 3.0)
    video_turbo_steps: int = 8
    video_turbo_sampler: str = "euler"
    video_turbo_sched: str = "simple"
    video_noturbo_loras: tuple = ()
    video_noturbo_shift: tuple = (0.0, 3.0)
    video_noturbo_steps: int = 20
    video_noturbo_sampler: str = "res_multistep"
    video_noturbo_sched: str = "simple"

    # --- ComfyUI film-look catalogue entry (module-owned; film_extensions.md
    #     is left untouched -- the style node passes this text straight through) ---
    house_look_name: str = ""
    house_look_body: str = ""

    # ------------------------------------------------------------------ helpers ---
    @property
    def turbo_lora(self) -> str:
        return self.video_turbo_loras[0][0] if self.video_turbo_loras else ""

    @property
    def realism_lora(self) -> tuple:
        """(name, strength) of h3-realism-people in this style's turbo LoRA list,
        or ("", 0.0) when the style drops it (2d/3d/ghibli/anime all do; only
        photoreal keeps it)."""
        for ln, st in self.video_turbo_loras:
            if "realism" in ln:
                return (ln, st)
        return ("", 0.0)

    def apply_branch(self, base: dict) -> dict:
        """Copy of a BRANCHES entry with this style's video stack swapped in.
        h3-realism-people is dropped (it is simply not in the new lora list, so
        _graph_node's ``qual`` lookup and _graph_extend's ``realism`` lookup both
        come back empty and no realism LoRA node is emitted)."""
        b = dict(base)
        is_turbo = any("turbo" in ln for ln, _ in base.get("loras", []))
        if is_turbo:
            b["loras"]   = [list(t) for t in self.video_turbo_loras]
            b["shift"]   = tuple(self.video_turbo_shift)
            b["steps"]   = int(self.video_turbo_steps)
            b["sampler"] = self.video_turbo_sampler
            b["sched"]   = self.video_turbo_sched
        else:
            b["loras"]   = [list(t) for t in self.video_noturbo_loras]
            b["shift"]   = tuple(self.video_noturbo_shift)
            b["steps"]   = int(self.video_noturbo_steps)
            b["sampler"] = self.video_noturbo_sampler
            b["sched"]   = self.video_noturbo_sched
        return b

    def style_prompt(self, text: str) -> str:
        """Prepend the style block (and append the anti-pattern clause) to a drafted
        shot prompt. Idempotent-ish: if the block is already the first line, skip."""
        block = self.prompt_style_block.strip()
        body  = (text or "").strip()
        if body.startswith(block[:40]):
            return body
        parts = [block]
        if body:
            parts.append(body)
        if self.prompt_anti.strip():
            parts.append("STYLE RULES (obey, do not restyle the shot): "
                         + self.prompt_anti.strip())
        return "\n\n".join(parts)


# --------------------------------------------------------------------------------
# Local Stable Diffusion reference-plate graph (ComfyUI API prompt dict).
# Pure dict builder -- the caller (orch.py) runs it via its own run_graph() and
# picks up the SaveImage output. Kept here so both pipeline copies stay identical.
# --------------------------------------------------------------------------------
def sd_plate_graph(spec: "StyleSpec", prompt: str, width: int, height: int,
                   seed: int, out_prefix: str) -> dict:
    pos = (spec.plate_style_prefix.strip() + " " + (prompt or "").strip()).strip()
    neg = spec.plate_negative
    g = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": spec.plate_checkpoint}},
    }
    model, clip, vae = ["1", 0], ["1", 1], ["1", 2]
    nid = 10
    for ln, st in spec.plate_loras:
        g[str(nid)] = {"class_type": "LoraLoader", "inputs": {
            "model": model, "clip": clip, "lora_name": ln,
            "strength_model": float(st), "strength_clip": float(st)}}
        model, clip = [str(nid), 0], [str(nid), 1]
        nid += 1
    g["20"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": pos}}
    g["21"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": neg}}
    g["22"] = {"class_type": "EmptyLatentImage",
               "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
    g["30"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["20", 0], "negative": ["21", 0],
        "latent_image": ["22", 0], "seed": int(seed) & 0x7FFFFFFF,
        "steps": int(spec.plate_steps), "cfg": float(spec.plate_cfg),
        "sampler_name": spec.plate_sampler, "scheduler": spec.plate_scheduler,
        "denoise": 1.0}}
    g["40"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": vae}}
    g["50"] = {"class_type": "SaveImage",
               "inputs": {"images": ["40", 0], "filename_prefix": out_prefix}}
    return g
