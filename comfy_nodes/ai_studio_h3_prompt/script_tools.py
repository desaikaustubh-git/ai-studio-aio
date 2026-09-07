"""Script authoring + QA nodes for the AI Studio H3 lengthy-video pipeline.

Nodes, all category "AI Studio/H3":

  * AIStudioH3ScriptWriter  - a premise -> a tight, shot-by-shot video script that
        opens on a hook and keeps escalating. ChatGPT (default) or Ollama.
  * AIStudioH3StoryExpander - a SHORT story / premise -> a longer, more detailed
        shot-by-shot script that runs about a target duration (default 2.5 min),
        adding beats, locations, props and texture while keeping the source's
        premise, tone and ending. ChatGPT (default) or Ollama.
  * AIStudioH3Director      - a script -> a shot-by-shot DIRECTOR'S BREAKDOWN in the
        voice of Christopher Nolan: per shot the blocking, the props and where they
        sit, the camera setup, the lighting, the performance beats, the score and
        the sound design, plus a [GLOBAL LOOK] header. Feed shot_breakdown into
        AIStudioH3MultiShotRender's script. ChatGPT (default) or Ollama.
  * AIStudioH3ScriptReview  - runs the script past Gemini for logical + physical
        consistency (horror / fantasy may break physics on purpose) and
        restructures it so shot 1 is the hook, then 1-2 context shots, then the
        story. Outputs the revised script (feed it to AIStudioH3MultiShotRender)
        plus the review notes.
  * AIStudioH3RefImages     - reads the script's [GLOBAL ENTITIES] and generates one
        reference still per recurring character / location (Gemini or Flow / Nano
        Banana 2). Feed ref_images + ref_map into AIStudioH3MultiShotRender's ref2va
        mode so every shot renders the SAME guard, booth and figure.
  * AIStudioH3VideoReview   - samples frames across the finished video and has
        Gemini check them against the script: adherence, continuity,
        physics / artefacts.

The instruction text for each lives in system_prompts/{script_writer,
story_expander,director,script_review,video_review}.txt - edit those, no code
change. All reuse the browser / ollama plumbing from prompt_writer.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time

from .prompt_writer import (
    DEFAULT_CHATGPT_PYTHON,
    SYSTEM_PROMPT_DIR,
    _resolve_v2_dir,
    _run_chatgpt,
    _run_gemini,
    _run_image,
    _run_ollama,
    _run_qwen,
    _strip_reasoning,
)

GENRES = ["auto", "grounded", "horror", "sci-fi", "fantasy", "surreal"]

# Named feature-film "house looks" the Director / multi-shot planner can hold across
# a whole video (full blocks in system_prompts/film_extensions.md). Shared here so
# multi_shot.py imports one list. "auto" = pick from the material; "none" = general
# cinematography grammar only, no named look.
FILM_LOOKS = ["auto", "none", "Game of Thrones", "The Matrix", "TRON: Legacy",
              "Avatar", "The Odyssey", "Interstellar", "John Wick"]

_REVISED_MARKER = re.compile(r"={2,}\s*REVISED\s+SCRIPT\s*={2,}", re.I)
_REVIEW_HEADER = re.compile(r"^\s*={2,}\s*REVIEW\s*={2,}\s*", re.I)
_VERDICT_RE = re.compile(
    r"verdict\s*[:\-]\s*"
    r"(PASS WITH NOTES|PASS WITH FIXES|NEEDS WORK|PASS|FAIL)",
    re.I,
)
_SHOT_RE = re.compile(r"\[\s*SHOT\s+\d+\s*\]", re.I)
_SHOT_LINE_RE = re.compile(r"^\s*SHOT\s+\d+\s*[:\-]", re.I | re.M)


def _load_sys(filename: str, override: str) -> str:
    if override and override.strip():
        return override.strip()
    path = os.path.join(SYSTEM_PROMPT_DIR, filename)
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing system-prompt file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _extract_house_looks(text: str) -> str:
    """The '## 8. HOUSE LOOKS' slice of film_extensions.md, between the
    ``<!-- HOUSE-LOOKS:START -->`` / ``END`` markers. '' if absent."""
    m = re.search(r"<!--\s*HOUSE-LOOKS:START\s*-->(.*?)<!--\s*HOUSE-LOOKS:END\s*-->",
                  text or "", re.S)
    return m.group(1).strip() if m else ""


def _house_looks_catalogue(v2_dir: str = "") -> str:
    """The named feature-film 'house looks' from film_extensions.md - the
    user-editable ``<v2_dir>/output/`` copy first, then the bundled one."""
    cands = []
    if v2_dir:
        cands.append(os.path.join(v2_dir, "output", "film_extensions.md"))
    cands.append(os.path.join(SYSTEM_PROMPT_DIR, "film_extensions.md"))
    for p in cands:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    hl = _extract_house_looks(fh.read())
                if hl:
                    return hl
            except Exception:
                continue
    return ""


def _strip_fence(text: str) -> str:
    """Drop a single ``` fence wrapper the model may have added around a script."""
    text = _strip_reasoning(text or "").strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*)\n```\s*$", text, re.S)
    return m.group(1).strip() if m else text


def _has_shots(text: str) -> bool:
    return bool(_SHOT_RE.search(text or "") or _SHOT_LINE_RE.search(text or ""))


def _count_shots(text: str) -> int:
    return len(_SHOT_RE.findall(text or "")) or len(_SHOT_LINE_RE.findall(text or ""))


def _genre_line(genre: str) -> str:
    g = (genre or "auto").strip().lower()
    if g in ("", "auto"):
        return "GENRE hint: auto - infer the genre from the material itself."
    return f"GENRE hint: {g}."


def _sha(*parts) -> str:
    blob = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- Script Writer
class AIStudioH3ScriptWriter:
    """A premise -> a shot-by-shot script built to hook and escalate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "premise": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "The idea / logline / rough story for the video.",
                }),
                "shots": ("INT", {"default": 5, "min": 1, "max": 40}),
                "seconds_per_shot": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5}),
                "genre": (GENRES, {"default": "auto"}),
                "backend": (["chatgpt", "ollama"], {"default": "chatgpt"}),
            },
            "optional": {
                "title": ("STRING", {"default": ""}),
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
                "ollama_model": ("STRING", {"default": "qwen3:32b"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "chatgpt_python": ("STRING", {
                    "default": DEFAULT_CHATGPT_PYTHON,
                    "tooltip": "Interpreter with Playwright (ChatGPT + Gemini automation).",
                }),
                "v2_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Blank = use the copy bundled in this node's vendor/ folder.",
                }),
                "timeout_s": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF,
                                  "tooltip": "Bump to force a fresh call (identical inputs are cached)."}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script", "info")
    FUNCTION = "write"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, premise, shots, seconds_per_shot, genre, backend, title="",
                   extra_instructions="", ollama_model="", ollama_url="",
                   chatgpt_python="", v2_dir="", timeout_s=600, regen=0,
                   system_prompt_override="", **_):
        return _sha(premise, shots, seconds_per_shot, genre, backend, title,
                    extra_instructions, ollama_model, ollama_url, regen,
                    system_prompt_override)

    def write(self, premise, shots, seconds_per_shot, genre, backend, title="",
              extra_instructions="", ollama_model="qwen3:32b",
              ollama_url="http://127.0.0.1:11434",
              chatgpt_python=DEFAULT_CHATGPT_PYTHON, v2_dir="", timeout_s=600,
              regen=0, system_prompt_override=""):
        v2_dir = _resolve_v2_dir(v2_dir)
        premise = (premise or "").strip()
        if not premise:
            raise RuntimeError("premise is empty - describe the story you want.")

        system = _load_sys("script_writer.txt", system_prompt_override)
        req = [
            _genre_line(genre),
            f"Number of shots: exactly {int(shots)}.",
            f"Target duration per shot: about {float(seconds_per_shot):.1f} seconds.",
        ]
        if title.strip():
            req.append(f"Working title: {title.strip()}")
        req += ["", "PREMISE:", premise]
        if extra_instructions.strip():
            req += ["", "EXTRA INSTRUCTIONS:", extra_instructions.strip()]
        user = "\n".join(req)

        t0 = time.time()
        if backend == "ollama":
            raw = _run_ollama(system, user, ollama_model, ollama_url, timeout_s)
            label = f"ollama:{ollama_model}"
        else:
            raw = _run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s)
            label = "chatgpt (web)"
        elapsed = time.time() - t0

        script = _strip_fence(raw)
        if not _has_shots(script):
            raise RuntimeError(
                "script writer did not return a [SHOT N] script. Raw reply:\n"
                + script[:1200]
            )
        info = (f"backend={label} {elapsed:.0f}s "
                f"~{_count_shots(script)} shots {len(script)} chars")
        return {"ui": {"info": [info], "script": [script]},
                "result": (script, info)}


# --------------------------------------------------------------------- Story Expander
class AIStudioH3StoryExpander:
    """A short story / premise -> a longer, more detailed shot-by-shot script."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "short_story": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "The short story / premise / small shot list to expand.",
                }),
                "target_minutes": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.5,
                                             "tooltip": "Roughly how long the expanded video should run."}),
                "seconds_per_shot": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5,
                                               "tooltip": "Shot count = target_minutes*60 / this (capped at 40)."}),
                "genre": (GENRES, {"default": "auto"}),
                "backend": (["chatgpt", "ollama"], {"default": "chatgpt"}),
            },
            "optional": {
                "title": ("STRING", {"default": ""}),
                "extra_directions": ("STRING", {"multiline": True, "default": "",
                                                "tooltip": "New elements / constraints to fold in while expanding."}),
                "ollama_model": ("STRING", {"default": "qwen3:32b"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "chatgpt_python": ("STRING", {
                    "default": DEFAULT_CHATGPT_PYTHON,
                    "tooltip": "Interpreter with Playwright (ChatGPT + Gemini automation).",
                }),
                "v2_dir": ("STRING", {"default": "",
                                      "tooltip": "Blank = use the copy bundled in this node's vendor/ folder."}),
                "timeout_s": ("INT", {"default": 900, "min": 30, "max": 3600}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF,
                                  "tooltip": "Bump to force a fresh call (identical inputs are cached)."}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("story", "info")
    FUNCTION = "expand"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, short_story, target_minutes, seconds_per_shot, genre, backend,
                   title="", extra_directions="", ollama_model="", ollama_url="",
                   chatgpt_python="", v2_dir="", timeout_s=900, regen=0,
                   system_prompt_override="", **_):
        return _sha(short_story, target_minutes, seconds_per_shot, genre, backend,
                    title, extra_directions, ollama_model, ollama_url, regen,
                    system_prompt_override)

    def expand(self, short_story, target_minutes, seconds_per_shot, genre, backend,
               title="", extra_directions="", ollama_model="qwen3:32b",
               ollama_url="http://127.0.0.1:11434",
               chatgpt_python=DEFAULT_CHATGPT_PYTHON, v2_dir="", timeout_s=900,
               regen=0, system_prompt_override=""):
        v2_dir = _resolve_v2_dir(v2_dir)
        short_story = (short_story or "").strip()
        if not short_story:
            raise RuntimeError("short_story is empty - paste the short story to expand.")

        secs = max(1.0, float(seconds_per_shot))
        total_s = max(30.0, float(target_minutes) * 60.0)
        shots = max(4, min(40, round(total_s / secs)))

        system = _load_sys("story_expander.txt", system_prompt_override)
        req = [
            _genre_line(genre),
            f"Target total duration: about {float(target_minutes):.1f} minutes "
            f"(~{int(round(total_s))} seconds).",
            f"Target per-shot duration: about {secs:.1f} seconds.",
            f"Target number of shots: about {int(shots)} (a couple more or fewer is fine).",
        ]
        if title.strip():
            req.append(f"Working title: {title.strip()}")
        req += ["", "SHORT STORY TO EXPAND:", short_story]
        if extra_directions.strip():
            req += ["", "EXTRA DIRECTIONS (fold these new elements in):", extra_directions.strip()]
        user = "\n".join(req)

        t0 = time.time()
        if backend == "ollama":
            raw = _run_ollama(system, user, ollama_model, ollama_url, timeout_s)
            label = f"ollama:{ollama_model}"
        else:
            raw = _run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s)
            label = "chatgpt (web)"
        elapsed = time.time() - t0

        story = _strip_fence(raw)
        if not _has_shots(story):
            raise RuntimeError(
                "story expander did not return a [SHOT N] script. Raw reply:\n"
                + story[:1200]
            )
        n = _count_shots(story)
        info = (f"backend={label} {elapsed:.0f}s | target ~{int(shots)} shots / "
                f"{float(target_minutes):.1f} min | got ~{n} shots "
                f"(~{n * secs / 60.0:.1f} min) {len(story)} chars")
        return {"ui": {"info": [info], "story": [story]},
                "result": (story, info)}


# --------------------------------------------------------------------- Script Review
class AIStudioH3ScriptReview:
    """Gemini review + restructure: hook first, then context, then the story."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "The shot-by-shot script to review.",
                }),
                "genre": (GENRES, {"default": "auto"}),
                "backend": (["chatgpt", "gemini", "qwen", "ollama"], {"default": "chatgpt"}),
            },
            "optional": {
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
                "ollama_model": ("STRING", {"default": "qwen3:32b"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "chatgpt_python": ("STRING", {
                    "default": DEFAULT_CHATGPT_PYTHON,
                    "tooltip": "Interpreter with Playwright (ChatGPT + Gemini automation).",
                }),
                "v2_dir": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("revised_script", "review_notes", "info")
    FUNCTION = "review"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, script, genre, backend, extra_instructions="",
                   ollama_model="", ollama_url="", chatgpt_python="", v2_dir="",
                   timeout_s=600, regen=0, system_prompt_override="", **_):
        return _sha(script, genre, backend, extra_instructions, ollama_model,
                    ollama_url, regen, system_prompt_override)

    def review(self, script, genre, backend, extra_instructions="",
               ollama_model="qwen3:32b", ollama_url="http://127.0.0.1:11434",
               chatgpt_python=DEFAULT_CHATGPT_PYTHON, v2_dir="", timeout_s=600,
               regen=0, system_prompt_override=""):
        v2_dir = _resolve_v2_dir(v2_dir)
        script = (script or "").strip()
        if not script:
            raise RuntimeError("script is empty - wire in or paste the script to review.")

        system = (_load_sys("script_review.txt", system_prompt_override)
                  + "\n\n" + _genre_line(genre))
        user = script
        if extra_instructions.strip():
            user += "\n\nEXTRA INSTRUCTIONS:\n" + extra_instructions.strip()

        def _call(be):
            if be == "ollama":
                return (_run_ollama(system, user, ollama_model, ollama_url, timeout_s),
                        f"ollama:{ollama_model}")
            if be == "chatgpt":
                return (_run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s),
                        "chatgpt (web)")
            if be == "qwen":
                return (_run_qwen(system, user, chatgpt_python, v2_dir, timeout_s),
                        "qwen (web)")
            return (_run_gemini(system, user, chatgpt_python, v2_dir, timeout_s),
                    "gemini (web)")

        # No auto-fallback: the node uses exactly the backend you pick. ChatGPT is
        # the default (it wrote the script and does not refuse to review it);
        # Gemini reliably declines horror / violent creative content, so it is
        # opt-in only. If the chosen backend returns no usable [SHOT N] script the
        # loop below raises with the raw reply rather than silently trying another.
        chain = [backend]
        t0 = time.time()
        tried, reply, revised, notes, label = [], "", "", "", ""
        for be in chain:
            raw, label = _call(be)
            reply = _strip_reasoning(raw or "").strip()
            m = _REVISED_MARKER.search(reply)
            if m:
                notes = _REVIEW_HEADER.sub("", reply[:m.start()]).strip()
                revised = _strip_fence(reply[m.end():])
            else:
                notes = "(model did not emit a separate === REVIEW === section)"
                revised = _strip_fence(reply)
            tried.append(f"{label} {'ok' if _has_shots(revised) else 'no-script'}")
            if _has_shots(revised):
                break
        elapsed = time.time() - t0

        if not _has_shots(revised):
            raise RuntimeError(
                "script review did not return a revised [SHOT N] script "
                f"(tried: {', '.join(tried)}). Last raw reply:\n"
                + (reply[:1500] or "(empty)")
            )
        if len(tried) > 1:
            notes = (f"[{chain[0]} returned no usable script ({tried[0]}); "
                     f"fell back to {label}]\n\n" + notes)

        vm = _VERDICT_RE.search(notes) or _VERDICT_RE.search(reply)
        verdict = vm.group(1).upper() if vm else "REVIEWED"
        info = (f"backend={label} {elapsed:.0f}s verdict={verdict} "
                f"revised ~{_count_shots(revised)} shots {len(revised)} chars")
        return {"ui": {"info": [info], "review_notes": [notes or "(none)"],
                       "revised_script": [revised]},
                "result": (revised, notes, info)}


# --------------------------------------------------------------------- Video Review
def _sample_frames_to_disk(frames, n, max_w, out_dir):
    """frames: IMAGE tensor [T,H,W,C] float 0..1 -> up to n evenly-spaced JPEGs.

    Returns (list_of_paths, total_frame_count).
    """
    import numpy as np
    from PIL import Image

    total = int(frames.shape[0])
    if total == 0:
        raise RuntimeError("frames input is empty - nothing to review.")
    n = max(2, min(int(n), total))
    if n > 1:
        idxs = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    else:
        idxs = [0]

    paths = []
    for k, idx in enumerate(idxs):
        arr = frames[idx].detach().cpu().float().clamp(0, 1).numpy()
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = Image.fromarray(arr, "RGB")
        if img.width > int(max_w):
            new_h = max(1, round(img.height * int(max_w) / img.width))
            img = img.resize((int(max_w), new_h), Image.LANCZOS)
        pct = round(100 * idx / max(1, total - 1))
        p = os.path.join(out_dir, f"frame_{k:02d}_{pct:03d}pct.jpg")
        img.save(p, "JPEG", quality=88)
        paths.append(p)
    return paths, total


class AIStudioH3VideoReview:
    """Qwen Chat (default), Gemini or ChatGPT checks sampled video frames (or an
    mp4) against the script. No auto-fallback: whichever backend you pick is the
    only one contacted, so a decline never spills the frames/video to another
    service. On a decline the node returns verdict=DECLINED with an ACTION-NEEDED
    note and lets the graph continue (the render is already saved)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "script": ("STRING", {"multiline": True, "default": ""}),
                "genre": (GENRES, {"default": "auto"}),
                "backend": (["qwen", "gemini", "chatgpt"], {"default": "qwen"}),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. A real .mp4 to upload instead of sampled frames.",
                }),
                "num_probe_frames": ("INT", {"default": 12, "min": 2, "max": 32}),
                "max_probe_width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "qwen_model": ("STRING", {
                    "default": "Qwen3.7-Plus",
                    "tooltip": "backend=qwen only: label to pick in Qwen's model "
                               "dropdown. Blank = the site default.",
                }),
                "chatgpt_python": ("STRING", {
                    "default": DEFAULT_CHATGPT_PYTHON,
                    "tooltip": "Interpreter with Playwright (Qwen + ChatGPT + Gemini automation).",
                }),
                "v2_dir": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 600, "min": 60, "max": 3600}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "verdict")
    FUNCTION = "test"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, script, genre, backend="qwen", video_path="",
                   num_probe_frames=12, max_probe_width=1024, qwen_model="Qwen3.7-Plus",
                   chatgpt_python="", v2_dir="", timeout_s=600, regen=0,
                   system_prompt_override="", frames=None, **_):
        shape = tuple(getattr(frames, "shape", ()) or ())
        return _sha(script, genre, backend, video_path, num_probe_frames,
                    max_probe_width, qwen_model, regen, system_prompt_override, shape)

    def test(self, frames, script, genre, backend="qwen", video_path="",
             num_probe_frames=12, max_probe_width=1024, qwen_model="Qwen3.7-Plus",
             chatgpt_python=DEFAULT_CHATGPT_PYTHON,
             v2_dir="", timeout_s=600, regen=0, system_prompt_override=""):
        v2_dir = _resolve_v2_dir(v2_dir)
        script = (script or "").strip()
        if not script:
            raise RuntimeError("script is empty - wire in the script the video was made from.")

        work = tempfile.mkdtemp(prefix="h3_vreview_")
        vp = (video_path or "").strip()
        used_mp4 = bool(vp) and os.path.isfile(vp)
        if used_mp4:
            attach = [vp]
            probe_desc = f"the video file {os.path.basename(vp)}"
        else:
            attach, total = _sample_frames_to_disk(
                frames, num_probe_frames, max_probe_width, work)
            probe_desc = (
                f"{len(attach)} still frames sampled in order across the video "
                f"(each filename gives that frame's approximate % position; the "
                f"video is {total} frames long)"
            )

        system = (
            _load_sys("video_review.txt", system_prompt_override)
            + "\n\n" + _genre_line(genre)
            + f"\n\nYou are being given {probe_desc}."
        )
        user = "SCRIPT THE VIDEO WAS MADE FROM:\n\n" + script

        t0 = time.time()
        if backend == "qwen":
            raw = _run_qwen(system, user, chatgpt_python, v2_dir, timeout_s,
                            attach_paths=attach, model=qwen_model)
        elif backend == "gemini":
            raw = _run_gemini(system, user, chatgpt_python, v2_dir, timeout_s,
                              attach_paths=attach)
        else:
            raw = _run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s,
                               attach_paths=attach)
        elapsed = time.time() - t0

        report = _strip_reasoning(raw or "").strip()
        vm = _VERDICT_RE.search(report)
        if vm:
            verdict = vm.group(1).upper()
        elif not report:
            verdict, report = "DECLINED", f"({backend} returned no text.)"
        elif len(report) < 400:
            # short reply with no verdict = the model declined the content.
            verdict = "DECLINED"
        else:
            verdict = "UNKNOWN"
        if verdict in ("DECLINED", "NO REPLY"):
            verdict = "DECLINED"
            report += (
                f"\n\n---\nACTION NEEDED: {backend} declined to review this "
                f"content. Manual review is pending - check this render by hand "
                f"before publishing. The frames/video were NOT sent to any other "
                f"service."
            )
        header = (f"# {backend} video review | {elapsed:.0f}s | verdict={verdict} | "
                  f"probed {'mp4' if used_mp4 else str(len(attach)) + ' frames'}\n\n")
        out = header + report
        return {"ui": {"verdict": [verdict], "report": [out]},
                "result": (out, verdict)}


# --------------------------------------------------------------------- Ref Images
_ENTITY_RE = re.compile(
    r"^\[\s*(CHARACTER|ENVIRONMENT|LOCATION|PROP)\s*:\s*([^\]]+?)\s*\]\s*(.*)$",
    re.I,
)


def _parse_entities(script: str):
    """Pull [CHARACTER: X] / [ENVIRONMENT: X] / [PROP: X] blocks out of a script.

    Returns a list of {"kind", "name", "desc"} in file order, characters and
    environments before props. Description is the text after the tag plus any
    immediately-following indented / non-blank continuation lines.
    """
    lines = (script or "").splitlines()
    out = []
    i = 0
    while i < len(lines):
        m = _ENTITY_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        kind, name, first = m.group(1).upper(), m.group(2).strip(), m.group(3).strip()
        body = [first] if first else []
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith("[") or _SHOT_RE.match(nxt) or nxt.upper().startswith("[SHOT"):
                break
            body.append(nxt)
            j += 1
        desc = " ".join(x for x in body if x).strip()
        if name and desc:
            out.append({"kind": kind, "name": name, "desc": desc})
        i = j
    order = {"CHARACTER": 0, "ENVIRONMENT": 1, "LOCATION": 1, "PROP": 2}
    out.sort(key=lambda e: order.get(e["kind"], 3))
    return out


def _img_prompt_for(entity, style_suffix: str) -> str:
    kind, desc = entity["kind"], entity["desc"]
    if kind == "CHARACTER":
        # One clean full-figure identity plate (not a multi-panel grid - H3 tends to
        # render the grid). Lock the six things a character sheet locks (concept,
        # look, outfit, palette, style, pose) + negative notes, so later shots have
        # nothing left to drift on.
        framing = ("full-figure character reference of this one person, head to feet "
                   "in frame, standing relaxed and facing the camera at a slight "
                   "three-quarter angle, arms at the sides, neutral expression, even "
                   "soft frontal light, plain seamless mid-grey studio background, no "
                   "props, no other people. Lock the identity: exact face shape and "
                   "features, skin tone, hair colour and style, the complete outfit "
                   "and its colours, build and proportions, and any signature "
                   "accessory - keep every one of these identical in later shots; do "
                   "not restyle the hair, add or remove clothing, or change the "
                   "proportions")
    elif kind in ("ENVIRONMENT", "LOCATION"):
        framing = ("wide establishing interior view of the empty location, no people, "
                   "no characters, natural in-scene lighting, generous depth with a high "
                   "ceiling and a far wall or open doorway well back - an open, airy space "
                   "with room to move, never a small cramped boxed-in room")
    else:  # PROP
        framing = "product-style reference of the object, centred, plain grey background"
    return f"{desc}\n\n{framing}. {style_suffix}".strip()


def _load_cover(path, W, H):
    """Load an image, centre-crop to the W:H aspect, resize to WxH -> float [H,W,3]."""
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    src_ar, dst_ar = im.width / max(1, im.height), W / max(1, H)
    if src_ar > dst_ar:
        nw = max(1, int(round(im.height * dst_ar)))
        x = (im.width - nw) // 2
        im = im.crop((x, 0, x + nw, im.height))
    elif src_ar < dst_ar:
        nh = max(1, int(round(im.width / dst_ar)))
        y = (im.height - nh) // 2
        im = im.crop((0, y, im.width, y + nh))
    im = im.resize((int(W), int(H)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


class AIStudioH3RefImages:
    """Script [GLOBAL ENTITIES] -> one reference still per character / location."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": "",
                                      "placeholder": "The script - its [GLOBAL ENTITIES] block is read."}),
                "backend": (["gemini", "flow"], {"default": "gemini"}),
                "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 16}),
            },
            "optional": {
                "max_refs": ("INT", {"default": 4, "min": 1, "max": 9}),
                "include_props": ("BOOLEAN", {"default": False}),
                "entities_override": ("STRING", {"multiline": True, "default": "",
                                                 "tooltip": "One 'NAME | description' per line to use instead "
                                                            "of the script's [GLOBAL ENTITIES]."}),
                "style_suffix": ("STRING", {"multiline": True,
                                            "default": "photorealistic, sharp focus, even neutral lighting, "
                                                       "clean uncluttered reference image, no text, no watermark"}),
                "chatgpt_python": ("STRING", {"default": DEFAULT_CHATGPT_PYTHON,
                                              "tooltip": "Interpreter with Playwright."}),
                "v2_dir": ("STRING", {"default": ""}),
                "timeout_s": ("INT", {"default": 900, "min": 120, "max": 3600}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("ref_images", "ref_map", "info")
    FUNCTION = "make"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, script, backend, width, height, max_refs=4, include_props=False,
                   entities_override="", style_suffix="", chatgpt_python="", v2_dir="",
                   timeout_s=900, regen=0, **_):
        return _sha(script, backend, width, height, max_refs, include_props,
                    entities_override, style_suffix, regen)

    def _entities(self, script, entities_override, include_props, max_refs):
        if entities_override.strip():
            ents = []
            for ln in entities_override.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                name, _, desc = ln.partition("|")
                ents.append({"kind": "CHARACTER", "name": name.strip() or "ref",
                             "desc": (desc.strip() or name.strip())})
        else:
            ents = _parse_entities(script)
            if not include_props:
                ents = [e for e in ents if e["kind"] != "PROP"]
        if not ents:
            raise RuntimeError(
                "no [CHARACTER:] / [ENVIRONMENT:] blocks found in the script - fill "
                "entities_override, or check the script has a [GLOBAL ENTITIES] section.")
        return ents[:max(1, int(max_refs))]

    def make(self, script, backend, width, height, max_refs=4, include_props=False,
             entities_override="", style_suffix="", chatgpt_python=DEFAULT_CHATGPT_PYTHON,
             v2_dir="", timeout_s=900, regen=0):
        import torch

        v2_dir = _resolve_v2_dir(v2_dir)
        ents = self._entities(script, entities_override, include_props, max_refs)
        items = [{"name": e["name"], "prompt": _img_prompt_for(e, style_suffix)} for e in ents]

        out_dir = tempfile.mkdtemp(prefix="h3_refimg_")
        t0 = time.time()
        pairs = _run_image(backend, items, out_dir, chatgpt_python, v2_dir, timeout_s)
        elapsed = time.time() - t0

        got_names = {n for n, _ in pairs}
        kept = [e for e in ents if e["name"] in got_names]
        tensors = [torch.from_numpy(_load_cover(p, int(width), int(height)))
                   for _, p in pairs]
        ref_images = torch.stack(tensors, dim=0)  # [B,H,W,C]

        lines = []
        for k, e in enumerate(kept, 1):
            lines.append(f"<Picture {k}> = {e['name']} ({e['kind'].lower()}) - "
                         f"{e['desc'][:120]}")
        ref_map = "\n".join(lines)
        info = (f"backend={backend} {elapsed:.0f}s | {len(pairs)}/{len(items)} refs | "
                f"{int(width)}x{int(height)}")
        if len(pairs) < len(items):
            missing = [e["name"] for e in ents if e["name"] not in got_names]
            info += " | missing: " + ", ".join(missing)
        return {"ui": {"info": [info], "ref_map": [ref_map]},
                "result": (ref_images, ref_map, info)}


# --------------------------------------------------------------------- Director
class AIStudioH3Director:
    """A script -> a shot-by-shot director's breakdown, in the voice of Christopher
    Nolan: per shot, the blocking, the props and where they sit, the camera setup,
    the lighting, the performance beats, the score and the sound design. Feed the
    output into AIStudioH3MultiShotRender's ``script`` - the richer detail makes
    sharper H3 prompts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "The script / shot list to break down. [SHOT N] blocks "
                                   "are kept 1:1 when keep_shot_count is on.",
                }),
                "genre": (GENRES, {"default": "auto"}),
                "backend": (["chatgpt", "qwen", "ollama"], {"default": "chatgpt"}),
            },
            "optional": {
                "title": ("STRING", {"default": ""}),
                "look": ("STRING", {"multiline": True, "default":
                    "grounded, tactile, cinematic - one hard motivated light source, "
                    "everything else falling to black; a low patient score; dense layered "
                    "sound design with silence used as a weapon"}),
                "film_look": (FILM_LOOKS, {"default": "auto", "tooltip":
                    "Anchor the [GLOBAL LOOK] to a named feature-film 'house look' "
                    "(catalogue in system_prompts/film_extensions.md - Game of Thrones "
                    "/ The Matrix / TRON: Legacy / Avatar / The Odyssey / Interstellar "
                    "/ John Wick). 'auto' = the director picks the best fit for the "
                    "material; 'none' = no named look."}),
                "extra_directions": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Staging notes to fold in (a set piece, a prop, a camera idea)."}),
                "keep_shot_count": ("BOOLEAN", {"default": True,
                    "tooltip": "On: keep the script's [SHOT N] boundaries 1:1. Off: let the "
                               "director re-break the story into the shots it needs."}),
                "ollama_model": ("STRING", {"default": "qwen3:32b"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "chatgpt_python": ("STRING", {
                    "default": DEFAULT_CHATGPT_PYTHON,
                    "tooltip": "Interpreter with Playwright (ChatGPT + Gemini automation).",
                }),
                "v2_dir": ("STRING", {"default": "",
                                      "tooltip": "Blank = use the copy bundled in this node's vendor/ folder."}),
                "timeout_s": ("INT", {"default": 900, "min": 30, "max": 3600}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF,
                                  "tooltip": "Bump to force a fresh call (identical inputs are cached)."}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("shot_breakdown", "info")
    FUNCTION = "direct"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, script, genre, backend, title="", look="", film_look="auto",
                   extra_directions="", keep_shot_count=True, ollama_model="",
                   ollama_url="", chatgpt_python="", v2_dir="", timeout_s=900, regen=0,
                   system_prompt_override="", **_):
        return _sha(script, genre, backend, title, look, film_look, extra_directions,
                    keep_shot_count, ollama_model, ollama_url, regen,
                    system_prompt_override)

    def direct(self, script, genre, backend, title="", look="", film_look="auto",
               extra_directions="", keep_shot_count=True, ollama_model="qwen3:32b",
               ollama_url="http://127.0.0.1:11434",
               chatgpt_python=DEFAULT_CHATGPT_PYTHON, v2_dir="", timeout_s=900,
               regen=0, system_prompt_override=""):
        v2_dir = _resolve_v2_dir(v2_dir)
        script = (script or "").strip()
        if not script:
            raise RuntimeError("script is empty - paste the script to break down.")

        had_shots = _has_shots(script)
        n_in = _count_shots(script) if had_shots else 0
        system = _load_sys("director.txt", system_prompt_override)
        req = [_genre_line(genre)]
        if title.strip():
            req.append(f"Working title: {title.strip()}")
        if look.strip():
            req.append(f"LOOK the director should lean into: {look.strip()}")
        if had_shots and keep_shot_count:
            req.append(f"The script already has {n_in} [SHOT N] blocks - keep those exact "
                       f"boundaries, order and count, one output block per input shot.")
        elif not keep_shot_count:
            req.append("You MAY re-break the story into the shots it needs (about 5 s each, "
                       "one continuous take per shot).")

        fl = (film_look or "auto").strip()
        cat = _house_looks_catalogue(v2_dir) if fl != "none" else ""
        if cat:
            if fl not in ("auto", "none", ""):
                req.append(f'REFERENCE LOOK (required): cut this whole piece to the '
                           f'"{fl}" house look from the catalogue below. Put it on the '
                           f'[GLOBAL LOOK] "Reference look:" line and hold every shot\'s '
                           f'Camera, Lighting, palette and score to it.')
            else:
                req.append('REFERENCE LOOK: from the house-look catalogue below, pick '
                           'the single best-fitting feature-film look for this material '
                           '(a deliberate hybrid of two is allowed). Put it on the '
                           '[GLOBAL LOOK] "Reference look:" line and hold every shot to '
                           'it.')
            req += ["", "HOUSE-LOOK CATALOGUE (do not name any of these titles in the "
                    "breakdown - translate the look into concrete blocking, lens, light "
                    "and score):", cat]
        req += ["", "SCRIPT:", script]
        if extra_directions.strip():
            req += ["", "STAGING NOTES (fold these in):", extra_directions.strip()]
        user = "\n".join(req)

        if backend == "qwen":
            user += (
                "\n\nH3-FEASIBILITY PASS - this breakdown feeds the MiniMax-H3 "
                "image-to-video model. Restage each shot into something H3 renders "
                "cleanly WITHOUT changing the story, the beats, the dialogue or the "
                "shot count:\n"
                "- A transformation / reveal / vanish / substitution is a HARD CUT "
                "between two stable states - never a morph, melt or crossfade.\n"
                "- One simple camera move per shot (locked, or one slow push, or one "
                "slow pan) - no combined moves, no handheld, no rack focus; a "
                "'locked' / 'static' shot must be dead still.\n"
                "- One primary action per shot, one lit subject, deep black "
                "elsewhere. No crowds, no fine hand business, no fast limbs, no "
                "contact that needs real contact physics (dragging along the floor "
                "is fine, lifting / carrying is not).\n"
                "- Hands stay mostly out of frame or still; never a close-up of "
                "fingers doing detailed work.\n"
                "- Grade only what H3 holds: crushed black, hard thin rim light, "
                "matte surfaces; no bloom, haze, volumetrics or lifted blacks.\n"
                "- Each shot about 5 s of one continuous take.\n"
                "Return the SAME [SHOT N] blocks in the SAME order and count, only "
                "staged so H3 can hit them."
            )
        t0 = time.time()
        if backend == "ollama":
            raw = _run_ollama(system, user, ollama_model, ollama_url, timeout_s)
            label = f"ollama:{ollama_model}"
        elif backend == "qwen":
            raw = _run_qwen(system, user, chatgpt_python, v2_dir, timeout_s)
            label = "qwen (web)"
        else:
            raw = _run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s)
            label = "chatgpt (web)"
        elapsed = time.time() - t0

        breakdown = _strip_fence(raw)
        if not _has_shots(breakdown):
            raise RuntimeError(
                "director did not return a [SHOT N] breakdown. Raw reply:\n"
                + breakdown[:1200]
            )
        n_out = _count_shots(breakdown)
        has_look = "[GLOBAL LOOK]" in breakdown.upper()
        info = (f"backend={label} {elapsed:.0f}s | in {n_in or '?'} shots -> out {n_out} "
                f"shots | global-look {'yes' if has_look else 'no'} | {len(breakdown)} chars")
        return {"ui": {"info": [info], "shot_breakdown": [breakdown]},
                "result": (breakdown, info)}


NODE_CLASS_MAPPINGS = {
    "AIStudioH3ScriptWriter": AIStudioH3ScriptWriter,
    "AIStudioH3StoryExpander": AIStudioH3StoryExpander,
    "AIStudioH3Director": AIStudioH3Director,
    "AIStudioH3ScriptReview": AIStudioH3ScriptReview,
    "AIStudioH3RefImages": AIStudioH3RefImages,
    "AIStudioH3VideoReview": AIStudioH3VideoReview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioH3ScriptWriter": "H3 Script Writer (AI Studio)",
    "AIStudioH3StoryExpander": "H3 Story Expander (AI Studio)",
    "AIStudioH3Director": "H3 Director - Nolan (AI Studio)",
    "AIStudioH3ScriptReview": "H3 Script Review - Gemini (AI Studio)",
    "AIStudioH3RefImages": "H3 Reference Images (AI Studio)",
    "AIStudioH3VideoReview": "H3 Video Review - Gemini (AI Studio)",
}
