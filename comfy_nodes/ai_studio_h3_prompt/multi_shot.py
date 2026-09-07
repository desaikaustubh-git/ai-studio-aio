"""AIStudioH3MultiShotRender - turn one long script into one lengthy H3 video.

Given a full narrative, this node:

  1. Calls the LLM ONCE with the verbatim "H3 Prompt Formulas" rules (fl2va) plus a
     multi-shot wrapper, and gets back a JSON plan: an ordered list of shots, each
     with a complete standalone H3 prompt and a ``continuation`` flag.
  2. Renders the shots one by one, in order, inside this node - reusing the stock
     ComfyUI pipeline (MiniMaxH3ImageToVideo -> ConditioningZeroOut -> KSampler ->
     VAEDecode / VAEDecodeAudio). A shot marked ``continuation: true`` is seeded with
     the previous shot's final frame (fl2va first_frame), so it carries straight on;
     a shot marked ``continuation: false`` is a fresh text-to-video cut.
  3. Streams every shot to disk the moment it finishes - ``shot_NN.pt`` (raw
     frames) + a muxed ``shot_NN.mp4`` under ``<output>/multishot_shots/<ts>/``
     (``stream_shots`` on by default) - so only one shot is ever held in RAM.
     Then (``stream_assemble`` on by default, needs >=2 shots and no overlays)
     it joins the per-shot mp4s with the ffmpeg concat demuxer - video
     stream-copied, one continuous AAC track - into ``_full.mp4`` and returns
     its path on the ``final_video`` output; ``frames`` is then only a decimated
     preview. ``stream_assemble`` off (or overlays / <2 shots) = rebuild the
     whole continuous IMAGE tensor in RAM (~15 GB for a 50s 9:16 clip, which is
     what kept OOM-ing). ``stream_shots`` off = the old behaviour: keep every
     shot's frames in memory and ``torch.cat`` them at the end.

Reference images (``auto_refs``, on by default): before rendering, one still per
recurring element - character / location / prop / entity, taken from ``ref_entities``
or the script's ``[CHARACTER:]`` / ``[ENVIRONMENT:]`` / ``[PROP:]`` blocks - is
generated up front via the browser image provider (``ref_backend``, Gemini's
persistent chat by default, so the whole run shares one context). They are saved to
``image_out_dir`` and every shot is then anchored to them.

Two render paths:

* **fl2va + refs** (default): each shot rendered from its text prompt, conditioned
  on the reference stills via ``MiniMaxH3ReferenceToVideo`` so the recurring people
  and places stay consistent. ``continuation: true`` is ignored (every shot is a
  fresh cut). Wiring ``ref_images`` in from ``AIStudioH3RefImages`` still works and
  takes precedence over ``auto_refs``.
* **ref2va keyframes** (``shot_keyframes`` on, or ``mode = ref2va``): additionally
  generates a START and an END still for every shot (from the refs + that shot's
  beat) and renders the shot as a first<->last-frame interpolation via
  ``MiniMaxH3ImageToVideo``. A ``continuation: true`` shot skips its own start still
  and reuses the previous shot's final rendered frame as its first frame.

The single-shot ``AIStudioH3PromptWriter`` node is unchanged.

After rendering, an optional post pass composites a CCTV-style timestamp HUD onto
every frame (``hud_timestamp``), an engagement CTA caption over the first part of
the video (``cta_text`` / ``cta_fraction``), and a brand logo burned 100% opaque
into the bottom-right corner of every frame (``brand_logo`` / ``brand_logo_scale``
- covers any generator watermark). These are burned in with PIL rather than asked
of the model, so they are pixel-identical across shots; when any is set the planner
is also told to keep every shot free of on-screen text.

An optional turbo path (``speed_lora`` + ``sigma_shift``) swaps in a distilled
few-step LoRA and the shift-6 schedule it was trained for, cutting render time to
roughly a third. Steps still default to 8 (not 4): 4 is enough for the picture but
the generated audio is measurably worse. ``ddim_uniform`` is refused - on H3 it
discards the shot you asked for.

Staging / physical-truth pass (always on): the planner wrapper and an appended
"STAGING TRUTH" block force every shot to state a stage plan (camera + where each
person and object sits, relative to the furniture), an explicit in-scene eyeline
per person (never "at camera" unless the script says so), a beat sheet on the
one-per-second grid with any freeze as its own timed sentence, a continuity
capsule, and hard physical rules (solid bodies that go around furniture, real
contact and weight, stable anatomy, clean surfaces, its-own-restriction legible
text, two-figure contact staged close to a locked/pushed-in camera). These encode
the H3 failure modes seen in testing so the render gets the blocking right without
a review step.

Advanced film-making pass (``advanced_prompts``, on by default): a genuine SECOND
LLM call - the drafted @@@SHOT plan is handed back with the camera-move + film-look
libraries and each shot's wording is rewritten for a feature-film look and one
camera move, without touching structure, beats or staging. (Kept as its own pass
because the rule file + wrapper + a full director breakdown + two ~15 KB libraries
in one message overflows the ChatGPT web input.) It combines three sources
- the drafted shot, a CAMERA MOVE LIBRARY (``system_prompts/camera_moves.md``,
canonical copy ``output/sample_prompts.md``) and FILM LOOK EXTENSIONS
(``system_prompts/film_extensions.md`` - cinematography building blocks + named
feature-film "house looks"). ``film_look`` forces one house look (Game of Thrones
/ The Matrix / TRON: Legacy / Avatar / The Odyssey / Interstellar / John Wick) or
``auto`` lets the planner pick from the script's ``[GLOBAL LOOK]``. If pass 2
fails, refuses or changes the shot count, the drafted plan is kept as-is (it
already carries the full staging pass), so no per-video prompt surgery is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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
from .script_tools import (
    FILM_LOOKS,
    _img_prompt_for,
    _load_cover,
    _parse_entities,
    _strip_fence,
)

MAX_SHOTS_CEILING = 40


def _lora_choices():
    """``[""] + everything under models/loras`` for the ``speed_lora`` dropdown.

    A real combo (vs a free-text field) stops the ComfyUI frontend's
    "missing models" scan from flagging the turbo LoRA filename on load.
    Falls back to just ``[""]`` if ``folder_paths`` isn't importable yet.
    """
    try:
        import folder_paths
        return [""] + list(folder_paths.get_filename_list("loras"))
    except Exception:
        return [""]


# ------------------------------------------------------ advanced film-making pass
# After the planner drafts each shot's H3 prompt it runs one more pass, combining
# THREE sources into one seamless prompt: (1) the drafted shot, (2) the CAMERA
# MOVE LIBRARY (system_prompts/camera_moves.md, canonical copy at
# output/sample_prompts.md), (3) the FILM LOOK EXTENSIONS
# (system_prompts/film_extensions.md - cinematography building blocks + named
# feature-film "house looks"). Goal: Hollywood-feature-grade shots with no manual
# prompt surgery. Toggled by ``advanced_prompts``; the house look is forced /
# auto-picked by ``film_look`` (FILM_LOOKS, imported from script_tools).

_LIB_CAP = 16000


def _read_capped(path, cap=_LIB_CAP):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            txt = fh.read().strip()
    except Exception:
        return ""
    return txt if len(txt) <= cap else txt[:cap].rstrip() + "\n[...]"


# ---------------------------------------------------- per-shot disk spool helpers
# stream_shots renders each shot straight to disk and frees it before the next
# one, instead of piling every shot's frame tensor in RAM until the final concat
# (768x1344x3 float32 x ~175f x N shots is many GB). shot_NN.pt is the lossless
# spool the full output is rebuilt from; shot_NN.mp4 is a muxed, inspectable copy.

def _ffmpeg_bin():
    """imageio-ffmpeg's bundled exe -> ffmpeg on PATH -> the miniforge build."""
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    import shutil as _sh
    p = _sh.which("ffmpeg")
    if p:
        return p
    for c in (r"E:\pinokio\bin\miniforge\Library\bin\ffmpeg.exe",
              r"E:\pinokio\bin\miniforge\Library\bin\ffmpeg"):
        if os.path.isfile(c):
            return c
    return ""


def _write_shot_mp4(path, images_u8, waveform, sample_rate, fps=24.0):
    """Mux one shot (uint8 [N,H,W,3] RGB + optional waveform) to an mp4.

    Frames are streamed to ffmpeg's stdin as rawvideo in small chunks (so a full
    shot's ~0.5 GB never sits in a pipe buffer); audio goes via a temp wav.
    Best effort: returns ``path`` on success, ``""`` on any failure - the
    ``.pt`` spool is the guarantee, this is just a convenience copy.
    """
    ff = _ffmpeg_bin()
    if not ff:
        return ""
    import subprocess
    import wave as _wave
    n, h, w, _c = images_u8.shape
    wav = None
    proc = None
    try:
        if waveform is not None and getattr(waveform, "numel", lambda: 0)():
            wv = waveform
            if wv.dim() == 3:
                wv = wv[0]
            wv = wv.transpose(0, 1).contiguous().clamp_(-1.0, 1.0)   # [samples, ch]
            ch = max(1, int(wv.shape[1]))
            pcm = (wv * 32767.0).round().to("cpu").short().numpy().tobytes()
            wav = path + ".tmp.wav"
            with _wave.open(wav, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate or 44100))
                wf.writeframes(pcm)
        cmd = [ff, "-y", "-loglevel", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{fps:g}", "-i", "-"]
        if wav:
            cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", "19", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        arr = images_u8.to("cpu").contiguous().numpy()
        step = 8
        for s in range(0, n, step):
            proc.stdin.write(arr[s:s + step].tobytes())
        proc.stdin.close()
        return path if (proc.wait() == 0 and os.path.isfile(path)) else ""
    except Exception:
        try:
            if proc is not None:
                proc.kill()
        except Exception:
            pass
        return ""
    finally:
        if wav and os.path.isfile(wav):
            try:
                os.remove(wav)
            except OSError:
                pass


def _concat_mp4s(out_path, mp4_paths, waveform=None, sample_rate=None, fps=24.0):
    """Join the per-shot mp4s (identical codec/params from _write_shot_mp4) into
    one file with the ffmpeg concat demuxer - video is stream-COPIED, so there is
    no re-encode and no giant frame tensor in RAM (this is the OOM-safe path the
    in-RAM rebuild kept blowing up on: 51s @ 768x1344 float32 is ~15 GB). If a
    ``waveform`` is given it is muxed as ONE continuous AAC track over the joined
    video, replacing the slightly-clipped per-shot audio. Returns out_path on
    success, "" on any failure."""
    ff = _ffmpeg_bin()
    paths = [p for p in (mp4_paths or []) if p and os.path.isfile(p)]
    if not ff or len(paths) < 2:
        return ""
    import subprocess
    import wave as _wave
    lst = out_path + ".concat.txt"
    wav = None
    try:
        with open(lst, "w", encoding="utf-8") as fh:
            for p in paths:
                fh.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))
        if waveform is not None and getattr(waveform, "numel", lambda: 0)():
            wv = waveform
            if wv.dim() == 3:
                wv = wv[0]
            wv = wv.transpose(0, 1).contiguous().clamp_(-1.0, 1.0)   # [samples, ch]
            ch = max(1, int(wv.shape[1]))
            pcm = (wv * 32767.0).round().to("cpu").short().numpy().tobytes()
            wav = out_path + ".full.wav"
            with _wave.open(wav, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate or 44100))
                wf.writeframes(pcm)
        cmd = [ff, "-y", "-loglevel", "error", "-nostdin",
               "-f", "concat", "-safe", "0", "-i", lst]
        if wav:
            cmd += ["-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            cmd += ["-c", "copy"]
        cmd += ["-movflags", "+faststart", out_path]
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
        return out_path if (rc == 0 and os.path.isfile(out_path)) else ""
    except Exception:
        return ""
    finally:
        for _f in (lst, wav):
            if _f and os.path.isfile(_f):
                try:
                    os.remove(_f)
                except OSError:
                    pass


def _resolve_ref_lib(explicit, v2_dir, out_name, bundled_name):
    """Path to a reference library: explicit widget path (abs or rel to v2_dir) ->
    ``<v2_dir>/output/<out_name>`` (user-editable) -> bundled
    ``system_prompts/<bundled_name>`` (relocatable). '' if nothing found."""
    c = (explicit or "").strip()
    if c:
        if os.path.isfile(c):
            return c
        if v2_dir and os.path.isfile(os.path.join(v2_dir, c)):
            return os.path.join(v2_dir, c)
    if v2_dir:
        p = os.path.join(v2_dir, "output", out_name)
        if os.path.isfile(p):
            return p
    p = os.path.join(SYSTEM_PROMPT_DIR, bundled_name)
    return p if os.path.isfile(p) else ""


def _advanced_enrichment_block(film_look, camera_moves_txt, film_ext_txt):
    """The 'combine three sources' instruction + the two libraries, appended to the
    planner system prompt. '' when nothing to add."""
    cm = (camera_moves_txt or "").strip()
    fx = (film_ext_txt or "").strip()
    if not cm and not fx:
        return ""

    fl = (film_look or "auto").strip()
    if fl not in ("auto", "none", ""):
        look_para = (
            f'B. LOOK. Use the "{fl}" house look from the FILM LOOK EXTENSIONS below '
            f'for the WHOLE video. Weave its lens / format, lighting, palette, grade, '
            f'atmosphere and texture into the visual description, and its score + '
            f'sound-design signature into overall_soundscape and non_diegetic_music. '
            f'Keep it identical across every shot.')
    elif fl == "none":
        look_para = (
            'B. LOOK. Do NOT impose a named house look. Still apply the general '
            'cinematography building blocks from the FILM LOOK EXTENSIONS below '
            '(format & capture, lens language, lighting, colour & grade, atmosphere, '
            'composition, score & sound) in service of the script\'s [GLOBAL LOOK].')
    else:
        look_para = (
            'B. LOOK. If the script\'s [GLOBAL LOOK] names a feature-film look, use '
            'that. Otherwise pick the SINGLE best-fitting house look from the FILM '
            'LOOK EXTENSIONS below for this material (a deliberate hybrid of two is '
            'allowed) and commit to it. Weave that look\'s lens / format, lighting, '
            'palette, grade, atmosphere and texture into the visual description, and '
            'its score + sound-design signature into overall_soundscape and '
            'non_diegetic_music. Use the SAME look across every shot of the video.')

    parts = ["""
==================================================
ADVANCED FILM-MAKING ENRICHMENT - RUN THIS AS THE FINAL PASS ON EVERY SHOT
==================================================
After a shot's prompt is drafted to the specification above, do ONE more pass on
it, combining THREE sources into ONE seamless prompt. Keep the exact @@@SHOT output
format and every rule already stated - this pass only enriches wording, never the
structure and never the beats.

  SOURCE 1 - the shot you just drafted (its story, blocking, dialogue, beats). This
             is the spine; never drop or contradict it.
  SOURCE 2 - the CAMERA MOVE LIBRARY below.
  SOURCE 3 - the FILM LOOK EXTENSIONS below.

A. CAMERA. Decide the one move this shot needs. Find the closest entry in the
   CAMERA MOVE LIBRARY and use its geometry and its "End" framing to write the
   shot's camera sentence - ONE natural lowercase clause inside the description,
   never a heading, never stacked keywords. The library's speed / size words
   ("smooth", "gradual", "quick", "constant", "slow") are NOT allowed in the
   prompt: convert them to the only four permitted phrases - "with small
   amplitude", "with large amplitude", "at slow speed", "at fast speed" - or leave
   them out. One move per shot; never combine two. For a LOCKED shot say it out
   loud ("the camera holds one fixed position, no pan, tilt, zoom or push"). For a
   shot where two or more figures make physical contact (a grab, a lift, a stab, a
   hand-off), do NOT choose a long-lens or zoom move to get close - H3 widens
   those out; use a locked camera staged close, or a physical dolly in / push
   past, and keep the figures in contact range from the first beat.

""" + look_para + """

C. BUDGET. These additions are WORDS and TEXTURE, not new beats. Do not add
   actions, people, props or events to fit the look or the move. Stay at or under
   one beat per second. No on-screen text, no camera-rig / lens / format / film
   names in the prompt itself (describe the RESULT - the compression, the streak,
   the grain, the colour), no cramped rooms in vertical delivery, each dialogue
   line once.

D. OUTPUT. One finished prompt per shot. Never mention "library", "extension",
   "house look", "source", a film's title, or that any combining happened. The
   reader sees only the scene.

E. PHYSICAL TRUTH & BLOCKING - CARRY IT THROUGH, NEVER SOFTEN IT. This pass adds
   look and camera language only. It must not weaken, shorten or drop any staging
   the drafted shot already states: keep every explicit eyeline, every "seen from
   behind" / "does not turn", every held-still sentence and its duration, every
   "goes around the chair on this path", every continuity capsule, every
   legible-text restriction, and every "solid bodies, nothing passes through"
   clause - verbatim or stronger. If a look or a move would fight the blocking (a
   move that swings a subject round to face camera, a key light that needs them
   turned), change the look or the move, not the blocking. The story on screen -
   who is where, facing what, touching what - always wins over the palette. Keep
   every sound cue at full strength: a loud continuous score stays loud, a
   stinger stays sharp, never soften either to "ambient". Keep the pace: do not
   add a fourth action to a shot that has three, and keep every held-still
   duration exactly. Keep each camera's stated height, angle and lens feel -
   enrich the wording, never flatten it back to a plain eye-level lock.
"""]
    if cm:
        parts.append("\n=== CAMERA MOVE LIBRARY ===\n" + cm + "\n")
    if fx:
        parts.append("\n=== FILM LOOK EXTENSIONS ===\n" + fx + "\n")
    return "".join(parts)


# System prompt for PASS 2 (the enrichment pass). The drafted @@@SHOT plan is the
# user message; _advanced_enrichment_block(...) is appended to this. Kept as its
# own small pass because fl2va.txt + the wrapper + a full director breakdown plus
# two ~15 KB libraries in ONE message overflows the ChatGPT web input.
_ENRICH_SYSTEM = """You are running the ADVANCED FILM-MAKING ENRICHMENT pass on an
already-drafted MiniMax H3 multi-shot plan. The user message is a series of @@@SHOT
blocks; each is already a complete H3 prompt with its story, blocking, eyelines,
dialogue and beats.

Rewrite each block so its WORDING carries a feature-film look and one deliberate
camera move, following the enrichment spec and libraries below. You may not change
the structure: keep the exact @@@SHOT header lines, the shot count, the order, the
continuation flags, the seconds, the number and content of the beats, and the
dialogue. Keep every staging detail the draft states - each explicit eyeline, each
"seen from behind" / "does not turn", each timed held-still sentence, each "goes
around the chair on this path", each continuity capsule, each legible-text
restriction, each "solid bodies, nothing passes through" clause - verbatim or
stronger. If a look or a move would fight the blocking, change the look or the
move, not the blocking. Keep every sound cue at full strength (a loud continuous
score stays loud, a stinger stays sharp - never "ambient"); keep every held-still
duration and never add a further action to a shot; keep each camera's stated
height, angle and lens feel - enrich the wording, do not flatten it to an
eye-level lock.

Output ONLY the rewritten @@@SHOT blocks, same header lines, nothing before or
after, no code fence, no commentary."""

# Appended to system_prompts/fl2va.txt. {sec} / {frames} / {lo} / {hi} filled in.
_WRAPPER = """

==================================================
MULTI-SHOT MODE - READ THIS LAST, IT WINS ON OUTPUT FORMAT
==================================================
Everything above is the H3 prompt specification. Apply ALL of it to EACH shot.

The user gives you ONE continuous video's full narrative. Do this:

1. Break the narrative into an ordered list of SHOTS. Target roughly {sec} seconds
   per shot (~{frames} frames at 24 fps); cut on natural beats. One shot = one
   unbroken camera take. Produce between {lo} and {hi} shots.
   If the narrative is ALREADY divided into explicit shots (lines like "[SHOT 2]"
   or "Shot 2:"), keep those exact shot boundaries, their order and their count
   (one plan entry per script shot), use each shot's stated Duration for
   "seconds", and take "continuation" from that shot's own camera / continuity
   notes rather than re-cutting the script.

2. For EACH shot set "continuation":
   * false -> a NEW camera setup: a cut, a new angle, a location or time jump, or
             the first shot. Rendered fresh from text only.
   * true  -> the SAME unbroken take continuing from the previous shot. It will be
             rendered STARTING FROM the previous shot's final frame, so its opening
             beat must match where the previous shot ended (same framing, same
             subject placement, same lighting) and then move forward.
   Shot 1 is ALWAYS "continuation": false.

3. For EACH shot write a COMPLETE, self-contained H3 prompt that fully obeys the
   specification above (every required field, in the required form). The renderer
   sees only this one shot's text - never mention "previous shot" / "as before";
   instead describe the carried-over state explicitly.

4. BUILD EACH SHOT THE WAY A DIRECTOR BLOCKS IT. Before writing a shot's prose,
   fix these four things, then make the prose STATE them - do not leave any of
   them for the model to guess, because whatever you leave open it renders
   differently on every seed:
   a. STAGE PLAN. Where the camera sits (its height, its distance, which wall or
      subject it faces) and where every person and every key object is - as
      positions relative to the furniture and the walls, not just "in the room".
      Name the floor under the subject, what is directly behind them, and the
      colour and material of the nearest large object. Nobody and nothing appears
      from nowhere: any figure the shot uses is in frame from the first beat, at a
      stated spot and a stated distance. If the script places a figure behind /
      beside / over someone, that figure is PHYSICALLY in the frame there - the
      only exception is a figure the script shows on a screen / monitor /
      reflection only, and then you must write exactly that.
   b. EYELINE MAP. Give every person in frame ONE explicit gaze target that exists
      in the scene - a screen, a doorway, another person, an object, a fixed point
      in the middle distance - and write it as a short clause ("his gaze stays on
      the monitor wall", "she looks at the blade in his hand"). A character NEVER
      looks into the lens, at the camera, or "at the viewer" unless the script
      literally has them address camera. When the script has someone facing away
      from the camera, say so plainly ("seen from behind, the back of his head to
      the lens, he does not turn") so the render cannot rotate them to face front.
   c. BEAT SHEET. List the shot's beats in order on the one-beat-per-second grid
      (a 5 s shot = 3 to 5 beats, never more). A HELD STATE - freezes, goes rigid,
      does not move, holds his breath, stays absolutely still - is its OWN beat
      and gets its own sentence with a rough duration ("he holds completely still
      for about two seconds, no turn, no step, shoulders locked, breath shallow").
      Never fold a freeze into another sentence: on its own line it survives the
      render, buried inside another beat it is dropped.
   d. CONTINUITY CAPSULE. Restate the fixed look of each recurring person, place
      and prop the shot shows - a few concrete words each (wardrobe, hair, the
      room's key geometry, a prop's state), every shot. The renderer has no memory
      of the other shots and will drift without it.

5. PHYSICAL TRUTH - HARD RULES, APPLIED TO EVERY SHOT, NEVER SOFTENED:
   - Bodies and objects are SOLID. Anyone who moves past furniture goes AROUND it
     on a path you state ("steps around the right side of the chair, one hand on
     the backrest, and lowers him onto the open floor"); no person and no object
     ever passes through a chair, a desk, a wall, a door or another body. Whenever
     someone crosses the room or handles a body, write the path and the contact.
   - Contact carries weight: a hand that lifts a body grips it and takes the load,
     the chair rocks and creaks, the body bends where a body bends, feet take the
     weight. No floating, no gliding, no snapping between poses.
   - Anatomy holds: five fingers per hand, hands stay attached, limbs keep their
     length, one head per person, faces keep the same features beat to beat.
   - Every surface in frame is CLEAN: no watermark, sparkle, generator mark, logo,
     interface chrome, caption, subtitle or timestamp, and no gibberish text on
     any screen, wall or object, unless the script puts it there. A screen feed
     shows the scene it is a feed of, not UI.
   - Any short piece of text the STORY needs the viewer to read (a label, a neck
     stamp, a sign) gets its own restriction sentence: held close and filling the
     centre of the frame, sharp and legible, the exact characters in quotes,
     appearing once, not duplicated, not warped.
   - MULTI-FIGURE CONTACT (a grab, a stab, a lift, a hand-off, a shove): stage the
     two figures ALREADY close together at a LOCKED camera or one that physically
     pushes in. H3 does not honour "long lens" / "tight close-up" prose when two
     people interact - it widens the frame out. To land the tight shot, move the
     camera in (dolly in / push past) or set a locked camera close, and keep the
     figures within contact range from the first beat.
   - NO TELEPORT / NO RE-STAGING WITHIN A SHOT. A figure or object that begins the
     shot at a stated position stays on ONE continuous track for the whole shot -
     it never jumps, snaps, slides or cuts to a new position. If it moves, it
     travels a path you name (from where, around what object, to where) at a human
     speed. A single character is never shown twice in the same frame.
   - COLOUR & MATERIAL HOLD. Name the real colour of every garment, skin tone and
     key object once - "matte black", "pale blue", "bare concrete grey", not
     "dark" - and that colour is identical for the whole shot. Coloured lighting
     does not change it: cyan light does not turn a black coat brown or a grey
     wall blue.
   - EMOTION IS PHYSICAL ACTION. Never write "terrified", "rigid with fear", "in
     shock" and stop. Spell the state as two to four film-able actions (eyes fix
     and widen, breath held, a hard swallow, knuckles whitening on the desk edge,
     a single tremor). When the face is not in frame, carry the whole state in the
     body - shoulders locking, hands gripping, a flinch, then stillness.
   - CAMERA HEIGHT & ANGLE ARE A CHOICE. "Locked-off" is not flat eye-level
     coverage. State a height (low, eye, high, overhead), an angle (straight or
     canted, and how far), a lens feel, and what the framing does for the beat -
     low and close reads as threat, high and wide as helplessness, dead-centre
     symmetry as wrongness. One deliberate setup per shot, never a default.
   - SOUND IS NOT OPTIONAL. Every shot names (a) a non-diegetic score - its
     register, its instrument, and what it is doing (a sustained low drone, a
     dissonant string cluster swelling, a sub-bass pulse) - and (b) a diegetic
     bed. On a horror or thriller beat the score is LOUD and continuous, not faint
     ambience, with a sharp percussive stinger on the shock; silence is used only
     as one deliberate cut.
   - LET THE BEAT BREATHE. At the shot length given, a shot carries ONE main
     action with held time before and after it. A third distinct action means the
     shot is overloaded: give it its own shot or cut it.

STRICT RULE - 9:16 / VERTICAL DELIVERY: if the target format is vertical (9:16,
portrait, Shorts / Reels / TikTok), you must NOT stage any shot in a small, cramped
or boxed-in room. The vertical crop removes the sides, so tight interiors read as
claustrophobic and cheap. Every interior must have real depth and air: a high
ceiling well above the subject, a far wall or open doorway several metres back,
clear headroom and floor space, and foreground-to-background separation. Prefer
deep or open settings - corridors seen down their length, large halls, lobbies,
warehouses, exteriors - and keep the camera far enough back that the space feels
open, never pressed against the subject.

OUTPUT FORMAT - follow it EXACTLY and output nothing else (no preamble, no ``` fence,
no JSON). For each shot: one header line starting with @@@SHOT, then that shot's full
H3 prompt on the lines beneath it, then one blank line before the next shot.

@@@SHOT 1 | continuation: false | seconds: 5
<the complete H3 prompt for shot 1 - every required field in the required form,
plain text over as many lines as you need; quotes, colons and punctuation are all
fine here because this is not JSON>

@@@SHOT 2 | continuation: true | seconds: 5
<the complete H3 prompt for shot 2>

Shot numbers start at 1 and increase by 1. "continuation" is false or true only
(shot 1 is always false). Do NOT wrap the output in JSON, quotes or a code fence.
NEVER ask the user a question and NEVER put any text before the first @@@SHOT line -
if anything is ambiguous, decide and proceed.
"""


def _load_fl2va_rules(override: str) -> str:
    if override.strip():
        return override.strip()
    path = os.path.join(SYSTEM_PROMPT_DIR, "fl2va.txt")
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing rule file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _load_ref2va_rules(override: str) -> str:
    if override.strip():
        return override.strip()
    path = os.path.join(SYSTEM_PROMPT_DIR, "ref2va.txt")
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing rule file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def _load_shot_review(override: str = "") -> str:
    if override.strip():
        return override.strip()
    path = os.path.join(SYSTEM_PROMPT_DIR, "shot_review.txt")
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing rule file: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


_SHOT_VERDICT_RE = re.compile(
    r"verdict\s*[:\-]\s*(PASS WITH NOTES|PASS WITH FIXES|NEEDS WORK|PASS|FAIL)",
    re.I,
)
_SHOT_ADH_RE = re.compile(r"adherence\s*[:\-]\s*([0-9]{1,3})", re.I)


def _run_shot_reviewer(backend, system, user, mp4_path, chatgpt_python,
                       v2_dir, timeout_s):
    """One backend call to review a single shot mp4. Returns raw reply text."""
    attach = [mp4_path]
    if backend == "gemini":
        return _run_gemini(system, user, chatgpt_python, v2_dir, timeout_s,
                           attach_paths=attach)
    if backend == "chatgpt":
        return _run_chatgpt(system, user, chatgpt_python, v2_dir, timeout_s,
                            attach_paths=attach)
    return _run_qwen(system, user, chatgpt_python, v2_dir, timeout_s,
                     attach_paths=attach)


# What "good enough to move on" means when parsing a shot review.
_SHOT_PASS_VERDICTS = ("PASS", "PASS WITH NOTES")


_SHOT_REVISE_SYS = (
    "You tighten ONE shot's text-to-video prompt for the MiniMax-H3 model. You "
    "are given the shot's CURRENT PROMPT and a QC review that lists concrete "
    "defects. Rewrite the prompt so a re-render fixes every defect the review "
    "raised, changing as little else as possible. Keep the same shot: same "
    "subject, location, action, wardrobe, camera intent and length. Fold the "
    "review's 'Prompt fixes' in as explicit positive and negative wording "
    "(e.g. add 'blacks crushed to pure 0, no bloom, no glow, no haze', 'matte "
    "wool, zero sheen, zero specular', 'exactly five fingers', 'camera locked "
    "on a rigid tripod, zero drift or pan', 'hard cut on a single frame, no "
    "crossfade or melt'). Do NOT propose post-production, compositing, After "
    "Effects or inpaint work - the only lever is this prompt. Output ONLY the "
    "revised prompt as plain text: no preamble, no commentary, no code fence, "
    "no '@@@SHOT' header."
)


# Appended to system_prompts/ref2va.txt. {refmap} / {lo} / {hi} filled in.
_REF_WRAPPER = """

==================================================
MULTI-SHOT REFERENCE MODE - READ THIS LAST, IT WINS ON OUTPUT FORMAT
==================================================
Everything above is the H3 full-reference (six-section) prompt specification. Apply
ALL of it to EACH shot.

You are given ONE video's full script AND a fixed set of REFERENCE IMAGES that lock
the recurring characters and locations. The references, in order, are:
{refmap}

Do this:

1. Split the script into an ordered list of SHOTS. If it already has [SHOT N] blocks,
   keep those exact boundaries, order, count and each shot's Duration. Produce between
   {lo} and {hi} shots.

2. For EACH shot write a COMPLETE six-section full-reference H3 prompt, in this order,
   nothing renamed: subject_definitions, summary, retention_analysis,
   detailed_description, overall_soundscape, non_diegetic_music.
   * In subject_definitions bind every reference the shot uses, reusing the SAME
     <Picture N> numbers as the list above, e.g.:
       <Subject 1> is the night guard from <Picture 1>, <features to follow...>.
       <Subject 2> is the security-booth interior from <Picture 2>, <features...>.
     Only define references this shot actually shows.
   * In detailed_description refer to those locked elements as <Subject N> so every
     shot renders the SAME people and places. Restate everything - the renderer sees
     only this one shot, never say "as before".

3. There is NO continuation / frame-chaining in reference mode. Every shot is a fresh
   cut: ALWAYS write "continuation: false".

4. BLOCK EVERY SHOT LIKE A DIRECTOR, inside detailed_description - state these, do
   not leave them for the model to guess:
   - STAGE PLAN: where the camera sits and where each <Subject N> and key object
     is, as positions relative to the furniture and walls; name the floor under
     the subject, what is directly behind them, and the nearest large object's
     colour and material; every figure the shot uses is present in frame from the
     first instant at a stated spot and distance. A figure the script places
     behind / beside / over someone is PHYSICALLY in frame there; a figure shown
     on a screen / reflection only must be written as exactly that.
   - EYELINE: give each person ONE explicit in-scene gaze target and write it. No
     one looks into the lens, at the camera, or "at the viewer" unless the script
     has them address camera; when a subject faces away, write "seen from behind,
     he does not turn".
   - BEATS: one per second (a 5 s shot = 3 to 5 beats). A held state - freezes,
     goes rigid, does not move - is its OWN sentence with a rough duration, never
     folded into another beat.
   - PHYSICAL TRUTH: bodies and objects are solid - a mover goes AROUND furniture
     on a stated path and nothing passes through a chair, desk, wall or body; real
     contact and weight; five fingers, stable faces; every surface clean of
     watermark / interface / stray text; any text meant to be read gets its own
     "held close, centre frame, sharp, exact characters in quotes, shown once"
     restriction.
   - MULTI-FIGURE CONTACT (grab, stab, lift, hand-off): stage the figures already
     close at a locked or pushed-in camera; do not rely on long-lens prose for a
     two-person close-up - H3 widens those out.
   - NO TELEPORT: a figure / object that starts at a stated spot stays on one
     continuous track - no jump or snap to a new position; if it moves it walks a
     named path; one character never appears twice in a frame.
   - COLOUR HOLDS: name the real colour of each garment / skin / key object once
     ("matte black", not "dark") and hold it all shot - coloured light does not
     change it.
   - EMOTION = ACTION: never "terrified" / "rigid with fear" alone - two to four
     film-able actions (eyes fix, breath stops, a swallow, knuckles whiten), in
     the body when the face is not seen.
   - CAMERA IS A CHOICE: state height (low / eye / high / overhead), angle
     (straight / canted), a lens feel and why - not a flat eye-level default.
   - SOUND EVERY SHOT: a non-diegetic score (register + instrument + what it is
     doing) and a diegetic bed; a horror beat = LOUD continuous score + a stinger
     on the shock; silence only as a deliberate cut.
   - ONE main action per shot plus held time - no overloading.

STRICT RULE - 9:16 / VERTICAL DELIVERY: if the target format is vertical (9:16,
portrait, Shorts / Reels / TikTok), never place a shot in a small, cramped or
boxed-in room. The vertical crop removes the sides, so tight interiors read as
claustrophobic and cheap. In detailed_description give every interior real depth
and air: a high ceiling well above the subject, a far wall or open doorway several
metres back, clear headroom and floor space, foreground-to-background separation.
Prefer deep or open settings - corridors seen down their length, large halls,
lobbies, warehouses, exteriors - and keep the camera back so the space never feels
pressed against the subject. Locked <Subject N> locations still apply; frame them
wide and deep rather than tight.

OUTPUT FORMAT - exactly this, output nothing else (no preamble, no ``` fence, no JSON).
For each shot: one header line starting with @@@SHOT, then that shot's full six-section
prompt beneath it, then one blank line before the next shot.

@@@SHOT 1 | continuation: false | seconds: 5
subject_definitions:
<Subject 1> is ...

summary:
...

non_diegetic_music:
...

@@@SHOT 2 | continuation: false | seconds: 5
subject_definitions:
...

Shot numbers start at 1 and increase by 1. NEVER ask a question and NEVER put any text
before the first @@@SHOT line.
"""


_SHOT_HDR = re.compile(r"(?im)^[ \t]*@{2,3}[ \t]*SHOT\b[ \t]*")


def _parse_plan(raw: str):
    """LLM reply -> list of {shot, continuation, seconds, h3_prompt}.

    Accepts the @@@SHOT delimited format the planner is asked for, or a JSON array
    (hand-written plan_override, or an older planner reply).
    """
    txt = _strip_reasoning(raw or "").strip()
    m = re.search(r"```(?:json|text)?\s*\n(.*?)```", txt, re.S)
    if m:
        txt = m.group(1).strip()

    items = None

    # 1) @@@SHOT <n> | continuation: <bool> | seconds: <n>   +  free-text body
    if _SHOT_HDR.search(txt):
        items = []
        for part in _SHOT_HDR.split(txt)[1:]:
            head, _, body = part.partition("\n")
            cont = re.search(r"continuation\s*[:=]\s*(true|yes|1)\b", head, re.I) is not None
            sm = re.search(r"seconds?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", head, re.I)
            secs = float(sm.group(1)) if sm else 0.0
            prompt = re.sub(r"\n?@{2,3}[ \t]*$", "", body.strip()).strip()
            if prompt:
                items.append({"continuation": cont, "seconds": secs, "h3_prompt": prompt})

    # 2) JSON array / {"shots": [...]} fallback
    if not items:
        lo, hi = txt.find("["), txt.rfind("]")
        if lo != -1 and hi > lo:
            try:
                data = json.loads(txt[lo:hi + 1])
                items = data if isinstance(data, list) else None
            except ValueError:
                items = None
    if not items:
        ob, cb = txt.find("{"), txt.rfind("}")
        if ob != -1 and cb > ob:
            try:
                obj = json.loads(txt[ob:cb + 1])
                if isinstance(obj, dict):
                    items = obj.get("shots") or obj.get("plan")
            except ValueError:
                items = None

    if not isinstance(items, list) or not items:
        raise RuntimeError(
            "multi-shot planner did not return shots (looked for @@@SHOT blocks and "
            "a JSON array). Raw reply:\n" + txt[:1500]
        )

    shots = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(f"shot {idx + 1} is not an object: {item!r}")
        prompt = str(item.get("h3_prompt") or item.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError(f"shot {idx + 1} has no h3_prompt. Raw item: {item!r}")
        cont = bool(item.get("continuation", False)) and idx > 0
        try:
            secs = float(item.get("seconds") or 0) or 0.0
        except (TypeError, ValueError):
            secs = 0.0
        shots.append({"shot": idx + 1, "continuation": cont,
                      "seconds": secs, "h3_prompt": prompt})
    return shots


# ------------------------------------------------------- reference / keyframe images
# Every run (auto_refs) first generates one still per recurring element - character,
# location, prop, entity - via the browser image provider (Gemini's persistent chat
# by default, so the whole run shares one context and the look stays consistent).
# fl2va then feeds those stills to MiniMaxH3ReferenceToVideo as anchors. ref2va /
# shot_keyframes goes further: it also generates a START and an END still for every
# shot and renders the shot as a first<->last-frame interpolation; a continuation
# shot skips its own start still and reuses the previous shot's final rendered frame.

_REF_STYLE = ("photorealistic, sharp focus, even neutral lighting, clean uncluttered "
              "reference image, single subject, no text, no watermark, no caption")


def _ref_items_from(script, ref_entities):
    """-> [{"name","kind","desc"}] from the ``ref_entities`` override (one
    ``kind | name | description`` per line) or, failing that, the script's
    [CHARACTER:/ENVIRONMENT:/PROP:] blocks. Props are included."""
    ents = []
    for ln in (ref_entities or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) >= 3:
            kind, name, desc = parts[0], parts[1], " | ".join(parts[2:]).strip()
        elif len(parts) == 2:
            kind, name, desc = parts[0], parts[0], parts[1]
        else:
            kind, name, desc = "prop", parts[0][:40], parts[0]
        k = kind.strip().lower()
        kind = ("CHARACTER" if k.startswith(("char", "person", "man", "woman", "guard"))
                else "ENVIRONMENT" if k.startswith(("env", "loc", "set", "booth", "room"))
                else "PROP" if k.startswith("prop")
                else "CHARACTER" if k.startswith(("entity", "creature", "monster", "figure"))
                else "PROP")
        if desc:
            ents.append({"kind": kind, "name": name or kind.lower(), "desc": desc})
    if not ents:
        ents = _parse_entities(script or "")
    seen, out = set(), []
    for e in ents:
        key = e["name"].lower()
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out[:9]


def _refs_to_tensor(pairs, W, H):
    """[(name, path)] -> (IMAGE tensor [B,H,W,3] or None, ref_map str, {name: path})."""
    import torch
    tensors, kept, by_name = [], [], {}
    for name, path in pairs:
        try:
            tensors.append(torch.from_numpy(_load_cover(path, int(W), int(H))))
            kept.append(name)
            by_name[name] = path
        except Exception:
            continue
    if not tensors:
        return None, "", {}
    ref_map = "\n".join(f"<Picture {k}> = {n}" for k, n in enumerate(kept, 1))
    return torch.stack(tensors, dim=0), ref_map, by_name


def _kf_prompt(kind, h3_prompt, ref_map):
    """One still that is the opening ('first') or closing ('last') frame of a shot,
    consistent with the refs already established in the image session."""
    when = ("the OPENING frame - the very first instant, before any motion"
            if kind == "first" else
            "the CLOSING frame - the very last instant, after the action resolves")
    body = re.sub(r"\s+", " ", (h3_prompt or "")).strip()[:1400]
    refs = (f" Keep every recurring character, location and prop identical to the "
            f"established references ({', '.join(ln.split('= ')[-1] for ln in ref_map.splitlines())})."
            if ref_map else "")
    return (f"Generate an image, photorealistic, vertical 9:16 aspect ratio: a single "
            f"still that is {when} of this shot.{refs}\n\nSHOT: {body}")


# --------------------------------------------------------------- post overlays
# A CCTV-style timestamp HUD and an engagement CTA caption, composited onto the
# finished frames AFTER rendering. Model-drawn on-screen text drifts shot to shot
# (the very thing the reference-image pass could not fully pin down); burning them
# in here makes them pixel-identical across every shot, with exact CTA timing.

def _load_font(names, size):
    from PIL import ImageFont
    win = os.environ.get("WINDIR", r"C:\Windows")
    dirs = [os.path.join(win, "Fonts"), "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts", "/Library/Fonts", "/System/Library/Fonts"]
    for n in names:
        for cand in [n] + [os.path.join(d, n) for d in dirs]:
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _fmt_clock(total_s, ampm):
    total_s = int(total_s) % 86400
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    if ampm:
        suf = "AM" if h < 12 else "PM"
        return f"{(h % 12) or 12:02d}:{m:02d}:{s:02d} {suf}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _wrap(text, font, draw, maxw):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if not cur or draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out or [text]


def _draw_hud(img, text, position, H, W):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    fs = max(14, int(H / 24))
    sw = max(1, fs // 12)
    font = _load_font(["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf",
                       "LiberationMono-Regular.ttf", "Menlo.ttc"], fs)
    margin = max(10, int(H * 0.04))
    b = d.textbbox((0, 0), text, font=font, stroke_width=sw)
    tw, th = b[2] - b[0], b[3] - b[1]
    x = margin if "left" in position else W - margin - tw
    y = margin if "top" in position else H - margin - th
    d.text((x - b[0], y - b[1]), text, font=font, fill=(245, 245, 245, 240),
           stroke_width=sw, stroke_fill=(0, 0, 0, 235))


def _hud_layer(W, H, text, position):
    from PIL import Image
    import numpy as np
    import torch
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _draw_hud(img, text, position, H, W)
    return torch.from_numpy((np.asarray(img, dtype=np.float32) / 255.0).copy())


def _cta_layer(W, H, text, position):
    from PIL import Image, ImageDraw
    import numpy as np
    import torch
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fs = max(18, int(H / 15))
    sw = max(1, fs // 14)
    font = _load_font(["ariblk.ttf", "arialbd.ttf", "segoeuib.ttf",
                       "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"], fs)
    lines = _wrap(text, font, d, int(W * 0.84))
    dims = [d.textbbox((0, 0), ln, font=font, stroke_width=sw) for ln in lines]
    ws = [q[2] - q[0] for q in dims]
    hs = [q[3] - q[1] for q in dims]
    gap = int(fs * 0.30)
    block_w, block_h = (max(ws) if ws else 0), sum(hs) + gap * (len(lines) - 1)
    padx, pady = int(fs * 0.8), int(fs * 0.55)
    margin = max(12, int(H * 0.055))
    bx0, bx1 = (W - block_w) // 2 - padx, (W + block_w) // 2 + padx
    if position == "top":
        by0 = margin
    elif position == "center":
        by0 = (H - block_h) // 2 - pady
    else:
        by0 = H - margin - block_h - 2 * pady
    by1 = by0 + block_h + 2 * pady
    d.rounded_rectangle([bx0, by0, bx1, by1],
                        radius=min((by1 - by0) // 2, int(fs * 1.1)),
                        fill=(0, 0, 0, 148))
    y = by0 + pady
    for ln, q, hh in zip(lines, dims, hs):
        d.text(((W - (q[2] - q[0])) // 2 - q[0], y - q[1]), ln, font=font,
               fill=(255, 255, 255, 255), stroke_width=sw, stroke_fill=(0, 0, 0, 190))
        y += hh + gap
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()), len(lines)


def _brand_layer(W, H, path, scale):
    """[H,W,4] overlay: the round logo badge in the bottom-right corner.

    A circular alpha is applied - only the disc shows, the square backing of the
    source PNG is dropped - and the disc itself is fully opaque so it sits over any
    generator watermark / sparkle and hides it.
    """
    from PIL import Image, ImageDraw
    import numpy as np
    import torch
    try:
        logo = Image.open(path).convert("RGBA")
    except Exception:
        return None
    # drop the flat square backing + its grey margin (badge disc ~= centre 81%)
    _w, _h = logo.size
    _ix, _iy = int(_w * 0.093), int(_h * 0.093)
    logo = logo.crop((_ix, _iy, _w - _ix, _h - _iy))
    lw = max(8, int(W * float(scale)))
    logo = logo.resize((lw, lw), Image.LANCZOS)
    ss = 4                                   # supersample the mask for a clean edge
    mask = Image.new("L", (lw * ss, lw * ss), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, lw * ss - 1, lw * ss - 1), fill=255)
    logo.putalpha(mask.resize((lw, lw), Image.LANCZOS))
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    margin = max(6, int(W * 0.025))
    canvas.paste(logo, (W - lw - margin, H - lw - margin), logo)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy())


def _blend(frames, layer, i0, i1, alpha_scale=None, chunk=24):
    """Alpha-composite an [H,W,4] overlay onto frames[i0:i1] ([N,H,W,3], 0..1), in place."""
    rgb = layer[..., :3].contiguous()
    a = layer[..., 3:4].contiguous()
    if alpha_scale is None:
        pre, inv = rgb * a, 1.0 - a
        for s in range(i0, i1, chunk):
            frames[s:min(i1, s + chunk)].mul_(inv).add_(pre)
    else:
        for s in range(i0, i1, chunk):
            e = min(i1, s + chunk)
            aa = a * alpha_scale[s - i0:e - i0].view(-1, 1, 1, 1)
            frames[s:e].mul_(1.0 - aa).add_(rgb * aa)


def _apply_overlays(frames, hud_timestamp, hud_position, hud_tick,
                    cta_text, cta_fraction, cta_position, fps=24.0,
                    brand_logo="", brand_logo_scale=0.16):
    import torch
    n, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    notes = []

    ts = (hud_timestamp or "").strip()
    if ts:
        mt = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?", ts)
        if hud_tick and mt:
            hh, mm_, ss = int(mt.group(1)), int(mt.group(2)), int(mt.group(3) or 0)
            ap = (mt.group(4) or "").upper()
            if ap == "PM" and hh < 12:
                hh += 12
            if ap == "AM" and hh == 12:
                hh = 0
            base = hh * 3600 + mm_ * 60 + ss
            i = 0
            while i < n:
                soff = int(i / fps)
                j = min(n, int((soff + 1) * fps))
                _blend(frames, _hud_layer(W, H, _fmt_clock(base + soff, bool(ap)),
                                          hud_position), i, j)
                i = j
            notes.append(f"hud '{ts}'->ticking {hud_position}")
        else:
            _blend(frames, _hud_layer(W, H, ts, hud_position), 0, n)
            notes.append(f"hud '{ts}' {hud_position}")

    cta = (cta_text or "").strip()
    if cta:
        cutoff = max(1, min(n, int(round(n * float(cta_fraction)))))
        layer, nlines = _cta_layer(W, H, cta.upper(), cta_position)
        asc = torch.ones(cutoff, dtype=frames.dtype)
        fin, fout = min(12, cutoff // 4), min(20, cutoff // 3)
        if fin:
            asc[:fin] = torch.linspace(0.0, 1.0, fin)
        if fout:
            asc[cutoff - fout:] = torch.linspace(1.0, 0.0, fout)
        _blend(frames, layer, 0, cutoff, alpha_scale=asc)
        notes.append(f"cta {nlines}L first {cutoff}f/~{cutoff / fps:.1f}s {cta_position}")

    bl = (brand_logo or "").strip()
    if bl and os.path.isfile(bl):
        layer = _brand_layer(W, H, bl, brand_logo_scale)
        if layer is not None:
            _blend(frames, layer, 0, n)
            notes.append(f"brand '{os.path.basename(bl)}' bottom-right {int(float(brand_logo_scale) * 100)}%W opaque")

    frames.clamp_(0.0, 1.0)
    return "  overlays: " + " | ".join(notes)


class AIStudioH3MultiShotRender:
    """One long script -> shot plan (LLM) -> shots rendered in order -> one video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "frames_per_shot": ("INT", {
                    "default": 73, "min": 5, "max": 3600, "step": 1,
                    "tooltip": "Frames per shot at 24 fps (snapped to the 17k+5 grid by "
                               "the H3 node). Wire this from the Pixaroma duration node. "
                               "Total video length = this x number of shots.",
                }),
                "script": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "The full narrative for one lengthy video. "
                                   "The planner splits it into shots.",
                }),
                "backend": (["chatgpt", "ollama"], {"default": "chatgpt"}),
            },
            "optional": {
                "seconds_per_shot": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 60.0, "step": 0.5,
                                               "tooltip": "Advisory pacing target handed to the planner."}),
                "max_shots": ("INT", {"default": 12, "min": 1, "max": MAX_SHOTS_CEILING,
                                      "tooltip": "Hard cap. The plan is truncated to this many shots."}),
                "min_shots": ("INT", {"default": 2, "min": 1, "max": MAX_SHOTS_CEILING}),
                "extra_instructions": ("STRING", {"multiline": True, "default": "",
                                                  "tooltip": "Appended to the script, e.g. "
                                                             "'9:16 vertical, hard cuts, no on-screen text'."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "0 = random base seed each run. Shot N uses base+N."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "res_multistep",
                                            "tooltip": "Any ComfyUI sampler name (Ep29 uses res_multistep)."}),
                "scheduler": ("STRING", {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ollama_model": ("STRING", {"default": "qwen3:32b"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "chatgpt_python": ("STRING", {"default": DEFAULT_CHATGPT_PYTHON}),
                "v2_dir": ("STRING", {"default": "",
                                      "tooltip": "Blank = use the copy bundled in this node's vendor/ folder."}),
                "timeout_s": ("INT", {"default": 300, "min": 30, "max": 1800,
                                      "tooltip": "Ceiling for the ONE planner LLM call."}),
                "regen": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF,
                                  "tooltip": "Bump to force a fresh plan + render."}),
                "plan_override": ("STRING", {"multiline": True, "default": "",
                                             "tooltip": "Paste a JSON shot array to skip the LLM and "
                                                        "render exactly that (hand-tuning / re-runs)."}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["auto", "fl2va", "ref2va"], {"default": "auto",
                         "tooltip": "auto = ref2va when ref_images is wired, else fl2va."}),
                "ref_images": ("IMAGE", {"tooltip": "Reference stills (from H3 Reference Images). "
                                                    "Switches this node to ref2va - every shot is "
                                                    "conditioned on these for character/location lock."}),
                "ref_map": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "The '<Picture N> = ...' lines from H3 Reference Images."}),
                "ref_image_size": (["match", "max"], {"default": "match",
                                   "tooltip": "'max' = 2048px refs for best identity, several times slower."}),
                "hud_timestamp": ("STRING", {"default": "", "tooltip":
                    "CCTV-style clock burned onto EVERY frame in post (e.g. '01:42 AM'). "
                    "Blank = off. Makes the timestamp pixel-identical in every shot."}),
                "hud_position": (["top-right", "top-left", "bottom-right", "bottom-left"],
                                 {"default": "top-right"}),
                "hud_tick": ("BOOLEAN", {"default": False, "tooltip":
                    "Advance the clock's seconds across the video (live-feed look). "
                    "Off = the exact string on every frame."}),
                "cta_text": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "Engagement caption composited over the FIRST part of the video, e.g. "
                    "'Hit the follow button, heart, comment and share this video'. Blank = off."}),
                "cta_fraction": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Portion of the video from the start that shows the CTA caption."}),
                "cta_position": (["bottom", "top", "center"], {"default": "bottom"}),
                "speed_lora": (_lora_choices(), {"default": "", "tooltip":
                    "Turbo / speed LoRA from models/loras (lightx2v Minimax-h3-Turbo, "
                    "e.g. minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy). Blank = "
                    "full-quality path. When set, steps drop to 8 and sampler to euler "
                    "unless you changed them, and sigma_shift defaults to 6."}),
                "speed_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "quality_lora": (_lora_choices(), {"default": "", "tooltip":
                    "Optional SECOND LoRA from models/loras, stacked on top of "
                    "speed_lora (e.g. h3-realism-people-t2v-i2v-r2v). Applied to "
                    "the model only, after the turbo LoRA and before sigma_shift. "
                    "Blank = off."}),
                "quality_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sigma_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "H3 video flow shift. 0 = model default (12). The turbo LoRA is "
                               "distilled for 6; when speed_lora is set and this is 0, 6 is used."}),
                "sigma_shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.5,
                    "tooltip": "Audio flow shift paired with sigma_shift (H3 default 3, keep it)."}),
                "brand_logo": ("STRING", {"default": "", "tooltip":
                    "Logo PNG burned into the bottom-right corner of EVERY frame in post, "
                    "100% opaque (covers any generator watermark / sparkle). Absolute path, "
                    "or a path relative to v2_dir. Blank = off."}),
                "brand_logo_scale": ("FLOAT", {"default": 0.16, "min": 0.04, "max": 0.5, "step": 0.01,
                    "tooltip": "Logo width as a fraction of the frame width."}),
                "auto_refs": ("BOOLEAN", {"default": True, "tooltip":
                    "Generate one reference still per recurring element (character / "
                    "location / prop / entity) before rendering, and feed them to every "
                    "shot as anchors. Skipped if ref_images is already wired in."}),
                "ref_backend": (["gemini", "flow"], {"default": "gemini", "tooltip":
                    "Image provider for the reference stills and per-shot keyframes. "
                    "'gemini' keeps one persistent chat so the whole run shares context."}),
                "ref_entities": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "One 'kind | name | description' per line (kind = character / "
                    "environment / prop / entity). Blank = read the script's "
                    "[CHARACTER:] / [ENVIRONMENT:] / [PROP:] blocks."}),
                "shot_keyframes": ("BOOLEAN", {"default": False, "tooltip":
                    "ref2va: also generate a START and an END still for every shot and "
                    "render each shot as a first<->last-frame interpolation. A "
                    "continuation shot reuses the previous shot's final frame as its "
                    "start. Slower (2 image gens per shot) but locks composition."}),
                "image_out_dir": ("STRING", {"default": "", "tooltip":
                    "Folder to save the reference stills + keyframes into. Blank = a "
                    "temp folder."}),
                "image_timeout_s": ("INT", {"default": 1200, "min": 120, "max": 7200,
                    "tooltip": "Ceiling for the whole image-generation subprocess."}),
                "advanced_prompts": ("BOOLEAN", {"default": True, "tooltip":
                    "After drafting each shot, run a final pass that folds the CAMERA "
                    "MOVE LIBRARY (system_prompts/camera_moves.md / output/"
                    "sample_prompts.md) and the FILM LOOK EXTENSIONS "
                    "(system_prompts/film_extensions.md) into the shot's prompt for "
                    "Hollywood-feature-grade shots. Off = plain planner output."}),
                "film_look": (FILM_LOOKS, {"default": "auto", "tooltip":
                    "Which feature-film 'house look' to hold across the whole video. "
                    "'auto' = the planner picks from the script's [GLOBAL LOOK] / the "
                    "material. 'none' = general cinematography grammar only, no named "
                    "look. Needs advanced_prompts on."}),
                "camera_moves_file": ("STRING", {"default": "", "tooltip":
                    "Override path to the camera-move library (abs or relative to "
                    "v2_dir). Blank = output/sample_prompts.md, then the bundled "
                    "system_prompts/camera_moves.md."}),
                "film_extensions_file": ("STRING", {"default": "", "tooltip":
                    "Override path to the film-look extensions library. Blank = "
                    "output/film_extensions.md, then bundled "
                    "system_prompts/film_extensions.md."}),
                "stream_shots": ("BOOLEAN", {"default": True, "tooltip":
                    "Render each shot straight to disk (shot_NN.pt raw frames + a "
                    "muxed shot_NN.mp4) and free it before the next shot, instead "
                    "of holding every shot's frames in RAM until the final concat. "
                    "The frames/audio output is rebuilt from the spooled shots, so "
                    "the downstream save node is unaffected. Off = old all-in-RAM "
                    "behaviour."}),
                "shot_out_dir": ("STRING", {"default": "", "tooltip":
                    "Where the per-shot files go when stream_shots is on. Blank = "
                    "<ComfyUI output>/multishot_shots/<timestamp>/. The shot_NN.mp4 "
                    "files are kept for inspection; the shot_NN.pt scratch is "
                    "deleted once the final output is assembled."}),
                "stream_assemble": ("BOOLEAN", {"default": True, "tooltip":
                    "stream_shots only, >=2 shots, no HUD/CTA/brand overlays: join "
                    "the per-shot mp4s with ffmpeg concat (video stream-copied, one "
                    "continuous audio track) into <shot_out_dir>/_full.mp4 and "
                    "return its path on the final_video output. The frames output "
                    "becomes a decimated preview (a downstream Save Mp4 then writes "
                    "only a short thumbnail - use final_video for the real file). "
                    "This is the OOM-safe assembly path; off = rebuild the whole "
                    "float32 tensor in RAM (~15 GB for a 50s 9:16 clip)."}),
                "render_only_shots": ("STRING", {"default": "", "tooltip":
                    "Comma list of 1-based shot numbers to (re)render, e.g. '3' "
                    "or '2,5'. Blank = render all. Shots NOT listed are reused "
                    "from the existing shot_NN.pt + shot_NN.mp4 in shot_out_dir "
                    "(the baseline run must have used keep_spool). Pair with "
                    "plan_override to swap one shot's prompt and re-render just "
                    "that shot, keeping the rest."}),
                "keep_spool": ("BOOLEAN", {"default": False, "tooltip":
                    "Keep each shot_NN.pt lossless scratch after assembly instead "
                    "of deleting it, so a later render_only_shots pass can reuse "
                    "the other shots without re-rendering. shot_NN.pt then also "
                    "carries that shot's waveform."}),
                "per_shot_review": ("BOOLEAN", {"default": False, "tooltip":
                    "After each shot renders, have an LLM watch THAT shot alone "
                    "against its prompt. On a fail it rewrites that shot's prompt "
                    "from the critique and re-renders the shot (up to "
                    "max_shot_retries) before moving to the next shot. Needs "
                    "stream_shots on. The stitched-video review still runs "
                    "downstream as the final sign-off."}),
                "per_shot_review_backend": (["qwen", "gemini", "chatgpt"],
                    {"default": "qwen", "tooltip":
                     "Which logged-in web LLM watches each shot mp4."}),
                "max_shot_retries": ("INT", {"default": 2, "min": 0, "max": 4,
                    "tooltip": "Most prompt-rewrite + re-render passes per shot "
                    "before accepting the best take and moving on."}),
                "per_shot_min_adherence": ("INT", {"default": 80, "min": 0, "max": 100,
                    "tooltip": "A shot passes when the review verdict is PASS / "
                    "PASS WITH NOTES and its adherence score is at least this."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("frames", "audio", "info", "final_video")
    FUNCTION = "run"
    CATEGORY = "AI Studio/H3"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, script, backend, seed=0, regen=0, plan_override="",
                   width=0, height=0, frames_per_shot=0, seconds_per_shot=0.0,
                   max_shots=12, min_shots=2, extra_instructions="", steps=20, cfg=1.0,
                   sampler_name="", scheduler="", denoise=1.0, ollama_model="",
                   ollama_url="", chatgpt_python="", v2_dir="", timeout_s=300,
                   system_prompt_override="", mode="auto", ref_map="",
                   ref_image_size="match", ref_images=None, hud_timestamp="",
                   hud_position="top-right", hud_tick=False, cta_text="",
                   cta_fraction=0.5, cta_position="bottom", speed_lora="",
                   speed_lora_strength=1.0, quality_lora="", quality_lora_strength=1.0,
                   sigma_shift=0.0, sigma_shift_audio=3.0,
                   brand_logo="", brand_logo_scale=0.16, auto_refs=True,
                   ref_backend="gemini", ref_entities="", shot_keyframes=False,
                   image_out_dir="", image_timeout_s=1200, advanced_prompts=True,
                   film_look="auto", camera_moves_file="", film_extensions_file="",
                   stream_shots=True, shot_out_dir="", stream_assemble=True,
                   render_only_shots="", keep_spool=False,
                   per_shot_review=False, per_shot_review_backend="qwen",
                   max_shot_retries=2, per_shot_min_adherence=80, **_):
        if int(seed) <= 0:
            return str(time.time())          # random base seed -> always re-render
        blob = "\x1f".join(str(x) for x in (
            script, backend, seed, regen, plan_override, width, height, frames_per_shot,
            seconds_per_shot, max_shots, min_shots, extra_instructions, steps, cfg,
            sampler_name, scheduler, denoise, ollama_model, ollama_url, system_prompt_override,
            mode, ref_map, ref_image_size, tuple(getattr(ref_images, "shape", ()) or ()),
            hud_timestamp, hud_position, hud_tick, cta_text, cta_fraction, cta_position,
            speed_lora, speed_lora_strength, quality_lora, quality_lora_strength,
            sigma_shift, sigma_shift_audio,
            brand_logo, brand_logo_scale, auto_refs, ref_backend, ref_entities,
            shot_keyframes, advanced_prompts, film_look, camera_moves_file,
            film_extensions_file,
            per_shot_review, per_shot_review_backend, max_shot_retries,
            per_shot_min_adherence, render_only_shots, keep_spool,
        ))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- planning
    def _plan(self, script, backend, seconds_per_shot, frames_per_shot, min_shots,
              max_shots, extra_instructions, chatgpt_python, v2_dir, timeout_s,
              ollama_model, ollama_url, system_prompt_override, plan_override,
              mode="fl2va", ref_map="", advanced_prompts=True, film_look="auto",
              camera_moves_file="", film_extensions_file=""):
        if plan_override.strip():
            return _parse_plan(plan_override)

        if mode == "ref2va":
            rules = _load_ref2va_rules(system_prompt_override)
            system = rules + _REF_WRAPPER.format(
                refmap=(ref_map or "").strip() or "(no reference map supplied)",
                lo=int(min_shots), hi=int(max_shots),
            )
        else:
            rules = _load_fl2va_rules(system_prompt_override)
            system = rules + _WRAPPER.format(
                sec=f"{seconds_per_shot:.1f}" if seconds_per_shot > 0 else "5.0",
                frames=int(frames_per_shot), lo=int(min_shots), hi=int(max_shots),
            )

        user = script.strip()
        if extra_instructions.strip():
            user += "\n\n" + extra_instructions.strip()
        user += ("\n\nOutput the @@@SHOT blocks now and nothing else. Do not ask "
                 "questions and do not add commentary.")

        def _once(sys_p, u):
            if backend == "ollama":
                return _run_ollama(sys_p, u, ollama_model, ollama_url, timeout_s)
            return _run_chatgpt(sys_p, u, chatgpt_python, v2_dir, timeout_s)

        # PASS 1 - draft the shot plan. The advanced film-making libraries are NOT
        # in this message: fl2va.txt + the wrapper + a long director breakdown is
        # already close to the ChatGPT web message ceiling, and folding two ~15 KB
        # libraries in on top pushed it over ("message too long"). The enrichment
        # is a genuine second pass on the drafted shots instead (PASS 2).
        try:
            shots = _parse_plan(_once(system, user))
        except RuntimeError:
            harder = (user + "\n\n---\nYour previous reply had no shot blocks. Do NOT "
                      "ask for clarification. Produce EVERY shot now, each starting "
                      "with a line: @@@SHOT <n> | continuation: <true|false> | seconds: <n>")
            shots = _parse_plan(_once(system, harder))

        self._adv_note = ""
        if advanced_prompts:
            cm_path = _resolve_ref_lib(camera_moves_file, v2_dir,
                                       "sample_prompts.md", "camera_moves.md")
            fx_path = _resolve_ref_lib(film_extensions_file, v2_dir,
                                       "film_extensions.md", "film_extensions.md")
            block = _advanced_enrichment_block(
                film_look, _read_capped(cm_path), _read_capped(fx_path))
            enriched = self._enrich(shots, block, _once) if block else None
            if enriched:
                shots = enriched
            if block:
                self._adv_note = (
                    f"advanced: film_look={film_look or 'auto'} | "
                    f"camera_moves={'yes' if cm_path else 'MISSING'} | "
                    f"film_extensions={'yes' if fx_path else 'MISSING'} | "
                    f"pass2={'ok' if enriched else 'kept draft'}")
        return shots

    def _enrich(self, shots, block, _once):
        """PASS 2: hand the drafted @@@SHOT plan back with the camera-move + film-
        look libraries and have the model rewrite each block's wording only.
        Returns the enriched shots, or None (keep the draft) if the pass fails,
        refuses, or changes the shot count. continuation / seconds are always
        carried over from the draft - the enrichment may not touch structure."""
        drafted = "\n\n".join(
            f"@@@SHOT {s['shot']} | continuation: "
            f"{'true' if s['continuation'] else 'false'} | "
            f"seconds: {s['seconds'] or 5:g}\n{s['h3_prompt']}"
            for s in shots)
        sys_p = _ENRICH_SYSTEM + "\n" + block
        u = ("DRAFTED SHOT PLAN - rewrite EVERY @@@SHOT block below. Keep the exact "
             "header lines, the shot count, the order, the continuation flags, the "
             "seconds, the beats, the dialogue and every staging detail the draft "
             "states. Apply the enrichment to the wording only. Output ONLY the "
             "@@@SHOT blocks, nothing else.\n\n" + drafted)
        try:
            out = _parse_plan(_once(sys_p, u))
        except RuntimeError:
            return None
        if len(out) != len(shots):
            return None
        for o, s in zip(out, shots):
            o["shot"] = s["shot"]
            o["continuation"] = s["continuation"]
            o["seconds"] = s["seconds"]
        return out

    # ---------------------------------------------------------------- render
    def run(self, model, clip, vae, audio_vae, width, height, frames_per_shot,
            script, backend, seconds_per_shot=5.0, max_shots=12, min_shots=2,
            extra_instructions="", seed=0, steps=20, cfg=1.0,
            sampler_name="res_multistep", scheduler="simple", denoise=1.0,
            ollama_model="qwen3:32b", ollama_url="http://127.0.0.1:11434",
            chatgpt_python=DEFAULT_CHATGPT_PYTHON, v2_dir="", timeout_s=300,
            regen=0, plan_override="", system_prompt_override="",
            mode="auto", ref_images=None, ref_map="", ref_image_size="match",
            hud_timestamp="", hud_position="top-right", hud_tick=False,
            cta_text="", cta_fraction=0.5, cta_position="bottom",
            speed_lora="", speed_lora_strength=1.0, quality_lora="",
            quality_lora_strength=1.0, sigma_shift=0.0,
            sigma_shift_audio=3.0, brand_logo="", brand_logo_scale=0.16,
            auto_refs=True, ref_backend="gemini", ref_entities="",
            shot_keyframes=False, image_out_dir="", image_timeout_s=1200,
            advanced_prompts=True, film_look="auto", camera_moves_file="",
            film_extensions_file="", stream_shots=True, shot_out_dir="",
            stream_assemble=True, render_only_shots="", keep_spool=False,
            per_shot_review=False,
            per_shot_review_backend="qwen", max_shot_retries=2,
            per_shot_min_adherence=80,
            style_prompt_block="", ref_style_override=""):

        import torch
        import nodes
        import comfy.model_management as mm
        import comfy.utils

        v2_dir = _resolve_v2_dir(v2_dir)
        script = (script or "").strip()
        if not script and not plan_override.strip():
            raise RuntimeError("script is empty - paste the full narrative for the video.")

        _brand = str(brand_logo or "").strip()
        if _brand and not os.path.isfile(_brand) and v2_dir:
            _cand = os.path.join(v2_dir, _brand)
            if os.path.isfile(_cand):
                _brand = _cand

        # want_kf  -> also make a start+end still per shot, render first<->last frame
        # gen_refs -> auto-generate one still per recurring element up front
        # use_ref  -> render shots through MiniMaxH3ReferenceToVideo (refs as anchors)
        want_kf = (mode == "ref2va") or bool(shot_keyframes)
        have_refs_wired = ref_images is not None
        _ents = _ref_items_from(script, ref_entities) if (auto_refs or want_kf) else []
        gen_refs = bool(auto_refs and _ents and not have_refs_wired)
        use_ref = (have_refs_wired or gen_refs) and not want_kf
        if mode == "ref2va" and not (have_refs_wired or gen_refs or want_kf):
            raise RuntimeError("mode=ref2va needs ref_images wired or auto_refs on.")
        plan_mode = "ref2va" if use_ref else "fl2va"
        if not (ref_map or "").strip() and _ents and (gen_refs or have_refs_wired):
            ref_map = "\n".join(f"<Picture {k}> = {e['name']}"
                                for k, e in enumerate(_ents, 1))

        max_shots = max(1, min(int(max_shots), MAX_SHOTS_CEILING))
        min_shots = max(1, min(int(min_shots), max_shots))

        ei = extra_instructions or ""
        # Staging truth - appended to the user message on every render so it holds
        # even when system_prompt_override replaces the wrapper. Encodes the H3
        # failure modes seen in testing: gaze drifting to camera, dropped freezes,
        # a scripted figure missing from frame, bodies clipping through furniture,
        # generator marks echoed onto screens, long-lens two-figure shots widening.
        _truth = (
            "STAGING TRUTH - apply to EVERY shot, do not soften: give each person in "
            "frame ONE explicit in-scene eyeline and state it; no character looks "
            "into the lens, at the camera or 'at the viewer' unless the script has "
            "them address camera, and when a subject faces away say 'seen from "
            "behind, does not turn'. A 'freeze' / 'goes rigid' / 'does not move' is "
            "its OWN sentence with a rough duration, never folded into another beat. "
            "Any figure the script places behind / beside / over someone is "
            "physically in the frame at that spot from the first beat (a figure on a "
            "screen or reflection only must be written as exactly that). Bodies and "
            "objects are solid: a mover goes AROUND furniture on a stated path and "
            "nothing passes through a chair, desk, wall, door or body; contact takes "
            "weight. Keep hands at five fingers and faces stable. Every screen and "
            "surface is clean of watermark, sparkle, UI, caption, timestamp and "
            "gibberish text unless the script puts it there; text the story needs "
            "read gets its own 'held close, centre frame, sharp, exact characters, "
            "shown once' restriction. Stage any two-figure contact beat (grab, stab, "
            "lift, hand-off) close to a LOCKED or physically pushed-in camera - not "
            "on long-lens prose, which H3 widens out. A figure or object that starts "
            "a shot at a stated position stays on ONE continuous track - it never "
            "jumps, snaps, slides or cuts to a new spot mid-shot; if it must move it "
            "walks a path you name (from where, around what, to where) at human "
            "speed, and a single character is never shown twice in one frame. Name "
            "the actual colour of every garment, skin tone and key object once "
            "('matte black', 'pale blue', not 'dark') and hold it for the whole shot "
            "- coloured light does not turn black to brown or grey to blue. Never "
            "leave an emotional state as a mood word ('terrified', 'rigid with "
            "fear'): spell it as two to four film-able actions (eyes fix and widen, "
            "breath stops, a hard swallow, knuckles whiten on the desk, one tremor) "
            "and carry it in the body when the face is not in frame. 'Locked-off' is "
            "not flat eye-level TV coverage: state a camera height (low / eye / high "
            "/ overhead), an angle (straight / canted), a lens feel and why the "
            "framing serves the beat - one deliberate setup per shot. SOUND IS NOT "
            "OPTIONAL: every shot states a non-diegetic score (its register, "
            "instrument and what it is doing) and a diegetic bed; on a horror or "
            "thriller beat the score is LOUD and continuous, not faint ambience, "
            "with a sharp stinger on the shock and silence used only as one "
            "deliberate cut. Let beats breathe: at the given shot length carry ONE "
            "main action plus held time on each side - a third distinct action means "
            "the shot is overloaded.")
        ei = (ei.rstrip() + "\n\n" + _truth) if ei.strip() else _truth
        if str(hud_timestamp).strip() or str(cta_text).strip() or _brand:
            _clean = ("ON-SCREEN TEXT: do NOT render any burned-in timestamp, clock, HUD, "
                      "caption, subtitle, lower-third, channel logo or watermark in ANY "
                      "shot - keep every feed visually clean; all overlays are added in post.")
            ei = (ei.rstrip() + "\n\n" + _clean) if ei.strip() else _clean
        if int(height) >= int(width):
            _vert = ("STRICT RULE - vertical 9:16 delivery: never stage a shot in a small, "
                     "cramped or boxed-in room. Give every interior real depth and air - a "
                     "high ceiling above the subject, a far wall or open doorway several "
                     "metres back, clear headroom and floor space, foreground-to-background "
                     "depth - or use open / deep locations (corridors seen down their length, "
                     "halls, lobbies, warehouses, exteriors). Keep the camera back so the "
                     "space never feels pressed against the subject.")
            ei = (ei.rstrip() + "\n\n" + _vert) if ei.strip() else _vert

        # ---- optional turbo / speed-LoRA path (Pixaroma Ep32 findings) ----------
        # A distilled few-step LoRA on the model + a video sigma shift of 6 (not
        # the H3 default 12, which is what it was trained against) cuts render time
        # to roughly a third. Steps still want 8, not 4: 4 is enough for the
        # picture but the GENERATED audio is worse (blind test ranked 12>8>6>4).
        # euler/simple is the tested sampler pair; ddim_uniform throws the shot
        # away on every H3 model and is refused.
        eff_model, eff_steps = model, int(steps)
        eff_sampler = sampler_name or "res_multistep"
        eff_scheduler = scheduler or "simple"
        turbo_notes = []
        if "ddim_uniform" in (str(scheduler).strip(), str(sampler_name).strip()):
            raise RuntimeError(
                "ddim_uniform discards the shot you asked for on every H3 model "
                "(Pixaroma Ep32: 1.5/10 - replaces the scene, ~2x file size). "
                "Use scheduler 'simple', 'sgm_uniform' or 'beta'.")
        _sl = "" if str(speed_lora).strip().lower() in ("", "none") else str(speed_lora).strip()
        if _sl:
            import folder_paths
            try:
                _avail = list(folder_paths.get_filename_list("loras"))
            except Exception:
                _avail = []
            _name = _sl
            if _name not in _avail:
                _norm = _sl.replace("\\", "/").lower()
                _hit = [a for a in _avail
                        if a.replace("\\", "/").lower() == _norm
                        or os.path.basename(a).lower() == os.path.basename(_sl).lower()]
                if not _hit:
                    raise RuntimeError(
                        f"speed_lora '{_sl}' not in models/loras. "
                        + (f"Have: {', '.join(_avail[:10])}" if _avail else "loras folder is empty."))
                _name = _hit[0]
            _ln = nodes.NODE_CLASS_MAPPINGS.get("LoraLoaderModelOnly")
            if _ln is None:
                raise RuntimeError("LoraLoaderModelOnly node missing - update ComfyUI.")
            eff_model = _ln().load_lora_model_only(model, _name, float(speed_lora_strength))[0]
            turbo_notes.append(f"lora {os.path.basename(_name)}@{float(speed_lora_strength):g}")
            if int(steps) == 20:                       # factory default -> turbo sweet spot
                eff_steps = 8
                turbo_notes.append("steps 8")
            if (sampler_name or "res_multistep") == "res_multistep":
                eff_sampler = "euler"
                turbo_notes.append(f"{eff_sampler}/{eff_scheduler}")

        # optional second LoRA (quality / realism), stacked on eff_model after the
        # turbo LoRA and before the sigma shift.
        _ql = "" if str(quality_lora).strip().lower() in ("", "none") else str(quality_lora).strip()
        if _ql:
            import folder_paths
            try:
                _qavail = list(folder_paths.get_filename_list("loras"))
            except Exception:
                _qavail = []
            _qname = _ql
            if _qname not in _qavail:
                _qnorm = _ql.replace("\\", "/").lower()
                _qhit = [a for a in _qavail
                         if a.replace("\\", "/").lower() == _qnorm
                         or os.path.basename(a).lower() == os.path.basename(_ql).lower()]
                if not _qhit:
                    raise RuntimeError(
                        f"quality_lora '{_ql}' not in models/loras. "
                        + (f"Have: {', '.join(_qavail[:10])}" if _qavail else "loras folder is empty."))
                _qname = _qhit[0]
            _qln = nodes.NODE_CLASS_MAPPINGS.get("LoraLoaderModelOnly")
            if _qln is None:
                raise RuntimeError("LoraLoaderModelOnly node missing - update ComfyUI.")
            eff_model = _qln().load_lora_model_only(
                eff_model, _qname, float(quality_lora_strength))[0]
            turbo_notes.append(
                f"+lora {os.path.basename(_qname)}@{float(quality_lora_strength):g}")

        _shift = float(sigma_shift)
        if _sl and _shift <= 0.0:
            _shift = 6.0
        if _shift > 0.0:
            _ss = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3SigmaShift")
            if _ss is None:
                raise RuntimeError("MiniMaxH3SigmaShift node missing - update ComfyUI / H3 core.")
            eff_model = _ss.execute(model=eff_model, shift_video=_shift,
                                    shift_audio=float(sigma_shift_audio))[0]
            turbo_notes.append(f"shift {_shift:g}/{float(sigma_shift_audio):g}")

        t0 = time.time()
        shots = self._plan(script, backend, seconds_per_shot, frames_per_shot,
                           min_shots, max_shots, ei, chatgpt_python,
                           v2_dir, timeout_s, ollama_model, ollama_url,
                           system_prompt_override, plan_override,
                           plan_mode, ref_map, advanced_prompts, film_look,
                           camera_moves_file, film_extensions_file)
        if len(shots) > max_shots:
            shots = shots[:max_shots]
        shots[0]["continuation"] = False
        # animation-style pass (opt-in, driven by the AI Studio/H3/Style nodes):
        # fold the style block into every shot prompt AFTER planning/enrichment so
        # it also lands in the persisted prompt_used.txt and the keyframe prompts.
        # Empty string (every normal workflow) => untouched.
        _sblk = (style_prompt_block or "").strip()
        if _sblk:
            _head = _sblk[:40]
            for _s in shots:
                _bd = (_s.get("h3_prompt") or "").strip()
                _s["h3_prompt"] = _bd if _bd.startswith(_head) else f"{_sblk}\n\n{_bd}"
        plan_secs = time.time() - t0

        # ---- reference stills + (ref2va) per-shot start/end keyframes ----------
        img_note = ""
        kf_start = [None] * len(shots)
        kf_end = [None] * len(shots)
        if gen_refs or want_kf:
            out_dir = ((image_out_dir or "").strip()
                       or os.path.join(tempfile.gettempdir(), f"h3_refs_{int(time.time())}"))
            os.makedirs(out_dir, exist_ok=True)
            reused = []
            if gen_refs and not int(regen):
                # a prior run already dropped ref_*.png here - reuse them instead of
                # re-generating (iterating on the script shouldn't re-hit the provider)
                have = sorted(f for f in os.listdir(out_dir)
                              if re.match(r"ref_\d+_ref__.*\.(png|jpg|jpeg|webp)$", f, re.I))
                if len(have) >= max(1, len(_ents)):
                    reused = [(os.path.splitext(f)[0].split("ref__", 1)[-1],
                               os.path.join(out_dir, f)) for f in have]
                    gen_refs = False
            items, tags = [], []
            if gen_refs:
                for e in _ents:
                    items.append({"name": f"ref__{e['name']}",
                                  "prompt": _img_prompt_for(e, (ref_style_override or "").strip() or _REF_STYLE)})
                    tags.append(("ref", None))
            if want_kf:
                for i, s in enumerate(shots):
                    if not (bool(s["continuation"]) and i > 0):
                        items.append({"name": f"s{i + 1:02d}_start",
                                      "prompt": _kf_prompt("first", s["h3_prompt"], ref_map)})
                        tags.append(("start", i))
                for i, s in enumerate(shots):
                    items.append({"name": f"s{i + 1:02d}_end",
                                  "prompt": _kf_prompt("last", s["h3_prompt"], ref_map)})
                    tags.append(("end", i))
            img_secs = 0.0
            pairs = []
            if items:
                _ti = time.time()
                try:
                    pairs = _run_image(ref_backend, items, out_dir, chatgpt_python,
                                       v2_dir, int(image_timeout_s))
                except Exception as exc:
                    pairs, img_note = [], f"  images FAILED ({type(exc).__name__}: {exc})"
                img_secs = time.time() - _ti
            got = {n: p for n, p in pairs}
            ref_list = list(reused)
            for it, (kind, idx) in zip(items, tags):
                p = got.get(it["name"])
                if not p:
                    continue
                if kind == "ref":
                    ref_list.append((it["name"].split("ref__", 1)[-1], p))
                else:
                    t = torch.from_numpy(_load_cover(p, int(width), int(height)))[None]
                    (kf_start if kind == "start" else kf_end)[idx] = t
            if ref_list:
                rt, rmap, _bn = _refs_to_tensor(ref_list, int(width), int(height))
                if rt is not None:
                    ref_images, ref_map, have_refs_wired = rt, rmap, True
            n_kf = sum(x is not None for x in kf_start) + sum(x is not None for x in kf_end)
            if not img_note:
                _src = ("reused" if reused and not items else
                        f"{ref_backend} {img_secs:.0f}s")
                img_note = (f"  images: {_src} | {len(ref_list)} refs"
                            + (f" | {n_kf} keyframes" if want_kf else "") + f" -> {out_dir}")

        use_ref = use_ref and (ref_images is not None)
        if use_ref:
            for s in shots:
                s["continuation"] = False

        base_seed = int(seed) if int(seed) > 0 else random.randint(1, 2**31 - 1)

        h3_node = nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"]
        ref_node = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3ReferenceToVideo")
        zero_node = nodes.NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        decode_node = nodes.NODE_CLASS_MAPPINGS["VAEDecode"]()
        decode_audio = nodes.NODE_CLASS_MAPPINGS["VAEDecodeAudio"]

        ref_dict = None
        ref_len = int(frames_per_shot)
        if use_ref:
            if ref_node is None:
                raise RuntimeError("MiniMaxH3ReferenceToVideo node not found - update ComfyUI / H3 core.")
            rb = int(ref_images.shape[0])
            ref_dict = {f"ref_image_{k + 1}": ref_images[k:k + 1].contiguous()
                        for k in range(rb)}
            # ref2va's trained range is ~124-362 frames; snap up to the 17k+5 grid
            ref_len = max(124, int(frames_per_shot))
            while ref_len % 17 != 5:
                ref_len += 1

        # stream_shots: write each shot to disk as it finishes and free it, so
        # only one shot's frames are ever resident. shot_NN.pt is the lossless
        # spool the full IMAGE output is rebuilt from; shot_NN.mp4 is a muxed
        # copy for inspection. Off -> keep every shot in RAM and cat at the end.
        stream = bool(stream_shots)
        shot_dir = ""
        if stream:
            shot_dir = (shot_out_dir or "").strip()
            if not shot_dir:
                try:
                    import folder_paths
                    _base = folder_paths.get_output_directory()
                except Exception:
                    _base = os.path.join(v2_dir or tempfile.gettempdir(), "output")
                shot_dir = os.path.join(_base, "multishot_shots",
                                        time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(shot_dir, exist_ok=True)

        # persist the exact per-shot plan actually used (post-enrichment) so a
        # follow-up render_only_shots pass can pass it back as plan_override with
        # one shot's body swapped and re-render just that shot.
        if stream and shot_dir:
            try:
                with open(os.path.join(shot_dir, "plan.txt"), "w",
                          encoding="utf-8") as _pf:
                    for _s in shots:
                        _pf.write(
                            f"@@@SHOT {_s['shot']} | continuation: "
                            f"{'true' if _s['continuation'] else 'false'} | "
                            f"seconds: {(_s.get('seconds') or 5):g}\n"
                            f"{_s['h3_prompt']}\n\n")
            except OSError:
                pass

        # render_only_shots: 1-based shot numbers to (re)render; the rest are
        # reused from the spool (needs a prior keep_spool run). Empty = all.
        _only = set()
        for _tok in str(render_only_shots or "").replace(";", ",").split(","):
            _tok = _tok.strip()
            if _tok.isdigit():
                _only.add(int(_tok))
        _keep_pt = bool(keep_spool) or bool(_only)

        frame_chunks = []          # stream off: every shot's frames
        wave_chunks = []           # waveforms are tiny - always kept in RAM
        shot_pt = []               # stream on: spooled frame files
        shot_mp4 = []              # stream on: muxed per-shot copies ("" = failed)
        sample_rate = None
        prev_last = None
        pbar = comfy.utils.ProgressBar(len(shots))
        lengths = []

        # per_shot_review: after a shot renders, an LLM watches THAT shot alone
        # against its prompt; a fail rewrites the shot's prompt (from the
        # critique) and re-renders the shot, up to max_shot_retries, before the
        # loop moves on. Needs stream (the reviewer watches shot_NN.mp4).
        do_shot_review = bool(per_shot_review) and bool(stream)
        self._shot_reports = []     # full per-shot review text -> shot_reviews.txt
        shot_review_lines = []      # one compact verdict trail per shot

        for i, shot in enumerate(shots):
            mm.throw_exception_if_processing_interrupted()

            # reuse an untouched shot from the spool instead of re-rendering it
            if _only and (i + 1) not in _only and stream:
                _rpt = os.path.join(shot_dir, f"shot_{i + 1:02d}.pt")
                _rmp4 = os.path.join(shot_dir, f"shot_{i + 1:02d}.mp4")
                if not (os.path.isfile(_rpt) and os.path.isfile(_rmp4)):
                    raise RuntimeError(
                        f"render_only_shots={sorted(_only)} but shot {i + 1} has "
                        f"no spooled shot_{i + 1:02d}.pt / .mp4 in {shot_dir} - "
                        f"the baseline run must have used keep_spool.")
                _d = torch.load(_rpt, map_location="cpu")
                _u8r = _d["images"]
                _wr = _d.get("waveform")
                _srr = int(_d.get("sample_rate") or sample_rate or 44100)
                _nr = int(_u8r.shape[0])
                if _wr is None:
                    _wr = torch.zeros((1, 2, max(1, int(_nr / 24.0 * _srr))),
                                      dtype=torch.float32)
                wave_chunks.append(_wr)
                sample_rate = _srr
                lengths.append(_nr)
                prev_last = _u8r[-1:].to(torch.float32).div_(255.0)
                shot_pt.append(_rpt)
                shot_mp4.append(_rmp4)
                shot_review_lines.append(f"#{i + 1:>2} reused from spool")
                del _u8r, _d
                mm.soft_empty_cache()
                pbar.update(1)
                continue

            chained = bool(shot["continuation"]) and prev_last is not None

            def _render_take(cur_prompt, _i=i, _chained=chained):
                if want_kf and (kf_start[_i] is not None or kf_end[_i] is not None
                                or _chained):
                    _h3 = h3_node.execute(
                        clip=clip, vae=vae, prompt=cur_prompt,
                        width=int(width), height=int(height),
                        length=int(frames_per_shot),
                        first_frame=(prev_last if _chained else kf_start[_i]),
                        last_frame=kf_end[_i],
                    )
                elif use_ref:
                    _h3 = ref_node.execute(
                        clip=clip, vae=vae, audio_vae=audio_vae, prompt=cur_prompt,
                        width=int(width), height=int(height), length=ref_len,
                        ref_image_size=ref_image_size, ref_images=ref_dict,
                    )
                else:
                    _h3 = h3_node.execute(
                        clip=clip, vae=vae, prompt=cur_prompt,
                        width=int(width), height=int(height),
                        length=int(frames_per_shot),
                        first_frame=(prev_last if _chained else None),
                        last_frame=None,
                    )
                _pos, _lat = _h3[0], _h3[1]
                _neg = zero_node.zero_out(_pos)[0]
                _smp = nodes.common_ksampler(
                    eff_model, (base_seed + _i) % (2**63 - 1), int(eff_steps),
                    float(cfg), eff_sampler, eff_scheduler, _pos, _neg, _lat,
                    denoise=float(denoise),
                )[0]
                _img = decode_node.decode(vae, _smp)[0].to("cpu", torch.float32)
                _aud = decode_audio.execute(audio_vae, _smp)[0]
                _wav = _aud["waveform"].to("cpu", torch.float32)
                _sr = _aud["sample_rate"]
                del _smp, _pos, _neg, _lat, _h3, _aud
                mm.soft_empty_cache()
                return _img, _wav, _sr

            def _to_u8(_img):
                # slice the float32 -> uint8 cast; a one-shot `_img.mul(255.)`
                # allocates a second full-size buffer (~2.2 GB / 175f shot) and
                # OOM'd the CPU allocator on a loaded box.
                _step = 16
                _u8 = torch.empty_like(_img, dtype=torch.uint8)
                for _s in range(0, _img.shape[0], _step):
                    _blk = _img[_s:_s + _step]
                    _u8[_s:_s + _step] = (_blk.mul(255.0).round_()
                                          .clamp_(0.0, 255.0).to(torch.uint8))
                    del _blk
                return _u8

            cur_prompt = shot["h3_prompt"]
            images, wave_cpu, sr_cur = _render_take(cur_prompt)
            n_fr = int(images.shape[0])
            u8 = None
            mp4_path = ""
            if stream:
                u8 = _to_u8(images)
                del images
                images = None
                mp4_path = _write_shot_mp4(
                    os.path.join(shot_dir, f"shot_{i + 1:02d}.mp4"),
                    u8, wave_cpu, sr_cur, fps=24.0)

            if do_shot_review and mp4_path:
                trail = f"#{i + 1:>2}"
                for attempt in range(int(max_shot_retries) + 1):
                    try:
                        _raw = _run_shot_reviewer(
                            per_shot_review_backend,
                            _load_shot_review(system_prompt_override),
                            (f"This is SHOT {i + 1} of {len(shots)} from a longer "
                             f"film - review THIS shot alone. Target length "
                             f"~{shot.get('seconds', 5)}s.\n\nSHOT {i + 1} PROMPT "
                             f"/ INTENT:\n{cur_prompt}\n"),
                            mp4_path, chatgpt_python, v2_dir, int(timeout_s))
                    except Exception as exc:
                        trail += f" | a{attempt} review-ERR({type(exc).__name__})"
                        break
                    _rtxt = _strip_reasoning(_raw or "").strip()
                    _vm = _SHOT_VERDICT_RE.search(_rtxt)
                    _am = _SHOT_ADH_RE.search(_rtxt)
                    _vd = _vm.group(1).upper() if _vm else "FAIL"
                    _adh = int(_am.group(1)) if _am else 0
                    trail += f" | a{attempt} {_vd} {_adh}%"
                    self._shot_reports.append(
                        f"===== shot {i + 1} - attempt {attempt} - {_vd} {_adh}% "
                        f"=====\n{_rtxt}")
                    if (_vd in _SHOT_PASS_VERDICTS
                            and _adh >= int(per_shot_min_adherence)):
                        trail += " accept"
                        break
                    if attempt >= int(max_shot_retries):
                        trail += " (retries spent - keep last take)"
                        break
                    try:
                        _nraw = _run_chatgpt(
                            _SHOT_REVISE_SYS,
                            f"CURRENT PROMPT:\n{cur_prompt}\n\n----\nQC REVIEW:\n"
                            f"{_rtxt}\n",
                            chatgpt_python, v2_dir, int(timeout_s))
                        _np = _strip_fence(_strip_reasoning(_nraw or "")).strip()
                    except Exception as exc:
                        trail += f" revise-ERR({type(exc).__name__})-keep-last"
                        break
                    if not _np or _np == cur_prompt:
                        trail += " no-prompt-change-keep-last"
                        break
                    cur_prompt = _np
                    if u8 is not None:
                        del u8
                        u8 = None
                    mm.soft_empty_cache()
                    images, wave_cpu, sr_cur = _render_take(cur_prompt)
                    n_fr = int(images.shape[0])
                    u8 = _to_u8(images)
                    del images
                    images = None
                    mp4_path = _write_shot_mp4(
                        os.path.join(shot_dir, f"shot_{i + 1:02d}.mp4"),
                        u8, wave_cpu, sr_cur, fps=24.0)
                    trail += " -> rewrite+rerender"
                shot["h3_prompt"] = cur_prompt
                shot_review_lines.append(trail)

            wave_chunks.append(wave_cpu)
            sample_rate = sr_cur
            lengths.append(n_fr)

            if stream:
                prev_last = u8[-1:].to(torch.float32).div_(255.0)
                pt = os.path.join(shot_dir, f"shot_{i + 1:02d}.pt")
                torch.save({"images": u8, "waveform": wave_cpu,
                            "sample_rate": sr_cur}, pt)
                shot_pt.append(pt)
                shot_mp4.append(mp4_path)
                del u8
                u8 = None
            else:
                prev_last = images[-1:].clone()
                frame_chunks.append(images)
                images = None

            del wave_cpu
            mm.soft_empty_cache()
            pbar.update(1)

        if self._shot_reports and shot_dir:
            try:
                with open(os.path.join(shot_dir, "shot_reviews.txt"), "w",
                          encoding="utf-8") as _fh:
                    _fh.write("\n\n".join(self._shot_reports) + "\n")
            except OSError:
                pass

        # ---- one continuous audio track (waveforms are tiny, always in RAM) ---
        min_ch = min(w.shape[1] for w in wave_chunks)
        waveform = torch.cat([w[:, :min_ch, :] for w in wave_chunks], dim=-1)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}

        # ---- assemble the final video ----------------------------------------
        # stream_assemble: join the per-shot mp4s with ffmpeg (video copied, one
        # continuous AAC track) -> no 15 GB float32 rebuild. frames becomes a
        # decimated preview; final_video carries the real deliverable's path.
        final_video = ""
        _want_overlays = bool(str(hud_timestamp).strip() or str(cta_text).strip() or _brand)
        do_assemble = (stream and bool(stream_assemble) and len(shot_mp4) >= 2
                       and all(shot_mp4) and not _want_overlays)
        if do_assemble:
            final_video = _concat_mp4s(
                os.path.join(shot_dir, "_full.mp4"), shot_mp4,
                waveform=waveform, sample_rate=sample_rate, fps=24.0)
            if final_video:
                _keep = 96
                _tot = sum(lengths)
                _stride = max(1, (_tot + _keep - 1) // _keep)
                _acc = []
                for _pt in shot_pt:
                    _u8 = torch.load(_pt, map_location="cpu")["images"]
                    _acc.append(_u8[::_stride].clone())
                    del _u8
                frames = torch.cat(_acc, dim=0).to(torch.float32).div_(255.0)
                del _acc
            else:
                do_assemble = False          # concat failed -> full rebuild below

        if stream and not do_assemble:
            _first = torch.load(shot_pt[0], map_location="cpu")["images"]
            _, _H, _W, _C = _first.shape
            frames = torch.empty((sum(lengths), _H, _W, _C), dtype=torch.float32)
            _off = 0
            for _k, _pt in enumerate(shot_pt):
                _u8 = _first if _k == 0 else torch.load(_pt, map_location="cpu")["images"]
                _first = None
                _n = _u8.shape[0]
                frames[_off:_off + _n] = _u8.to(torch.float32).div_(255.0)
                _off += _n
                del _u8
        elif not stream:
            frames = torch.cat(frame_chunks, dim=0)
            frame_chunks.clear()

        overlay_line = ""
        if _want_overlays:
            try:
                overlay_line = _apply_overlays(
                    frames, hud_timestamp, hud_position, bool(hud_tick),
                    cta_text, cta_fraction, cta_position, fps=24.0,
                    brand_logo=_brand, brand_logo_scale=brand_logo_scale)
            except Exception as exc:
                overlay_line = f"  overlays FAILED ({type(exc).__name__}: {exc})"

        total = time.time() - t0
        chained_ns = [s["shot"] for s in shots if s["continuation"]]
        _nkf = sum(x is not None for x in kf_start) + sum(x is not None for x in kf_end)
        if want_kf and _nkf:
            mode_str = f"ref2va keyframes ({_nkf} stills, first<->last)"
        elif use_ref:
            mode_str = f"fl2va + refs ({ref_dict and len(ref_dict)} anchors, {ref_len}f/shot)"
        else:
            mode_str = "fl2va"
        _nf = sum(lengths)
        lines = [
            f"{len(shots)} shots | {_nf} frames "
            f"(~{_nf / 24.0:.1f}s @24fps) | {frames.shape[2]}x{frames.shape[1]} | {mode_str}",
            f"chained (continue from prev frame): {chained_ns or 'none'}",
            f"plan {plan_secs:.0f}s ({'override' if plan_override.strip() else backend}) | "
            f"render {total - plan_secs:.0f}s | base_seed {base_seed}",
        ]
        if turbo_notes:
            lines.append("  turbo: " + " | ".join(turbo_notes))
        else:
            lines.append(f"  sampler {eff_sampler}/{eff_scheduler} {eff_steps} steps")
        for s, n in zip(shots, lengths):
            tag = "CONT" if s["continuation"] else "cut "
            lines.append(f"  #{s['shot']:>2} {tag} {n:>4}f  {s['h3_prompt'][:90].replace(chr(10), ' ')}")
        if do_shot_review or shot_review_lines:
            if do_shot_review:
                lines.append(f"  per-shot review ({per_shot_review_backend}, "
                             f"<= {int(max_shot_retries)} retries, pass >= "
                             f"{int(per_shot_min_adherence)}%):")
            else:
                lines.append("  per-shot:")
            for _t in shot_review_lines:
                lines.append("    " + _t)
            if self._shot_reports:
                lines.append(f"    full reports -> {os.path.join(shot_dir, 'shot_reviews.txt')}")
        if getattr(self, "_adv_note", ""):
            lines.append("  " + self._adv_note)
        if img_note:
            lines.append(img_note)
        if overlay_line:
            lines.append(overlay_line)
        if stream:
            _kept = sum(1 for m in shot_mp4 if m)
            lines.append(f"  per-shot spool -> {shot_dir} "
                         f"({_kept}/{len(shots)} mp4)")
            if _only:
                lines.append(f"  render_only_shots: {sorted(_only)} "
                             f"(the rest reused from spool)")
            if _keep_pt:
                lines.append("  kept shot_NN.pt spool (keep_spool / render_only_shots)")
            if final_video:
                lines.append(f"  ASSEMBLED FULL VIDEO -> {final_video}  "
                             f"(frames output is a {frames.shape[0]}-frame preview only)")
            elif bool(stream_assemble):
                lines.append("  stream_assemble: concat unavailable "
                             "(missing per-shot mp4 / overlays / <2 shots) -> "
                             "rebuilt full tensor in RAM")
            if not _keep_pt:
                for _pt in shot_pt:                 # drop the lossless scratch
                    try:
                        os.remove(_pt)
                    except OSError:
                        pass
        info = "\n".join(lines)

        return {"ui": {"text": [info]},
                "result": (frames, audio_out, info, final_video)}


NODE_CLASS_MAPPINGS = {"AIStudioH3MultiShotRender": AIStudioH3MultiShotRender}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioH3MultiShotRender": "H3 Multi-Shot Lengthy Video (AI Studio)"
}
