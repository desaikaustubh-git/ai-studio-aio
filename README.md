# AI Studio ComfyUI — H3 Prompt Writer

Runs the **"H3 Prompt Formulas"** rule set (Ep29 → `Other Resources/H3 Prompt Formulas`)
*inside* ComfyUI. You type a plain description of the video you want; a custom node
calls an LLM with the H3 rule file as its system prompt and returns a finished
MiniMax H3 prompt, wired straight into the H3 video nodes. No copy‑paste from a
browser, and the whole thing is a normal ComfyUI graph you can edit in the front end.

```
[ H3 Prompt Writer ] --h3_prompt(STRING)--> [ MiniMax H3 …ToVideo ].prompt --> KSampler --> video
```

---

## What's in here

| Path | What it is |
|---|---|
| `comfy_nodes/ai_studio_h3_prompt/` | the custom nodes (`H3 Prompt Writer` + `H3 Multi-Shot Lengthy Video` + `H3 Script Writer` + `H3 Story Expander` + `H3 Director – Nolan` + `H3 Script Review – Gemini` + `H3 Reference Images` + `H3 Video Review – Gemini`, all *AI Studio*) |
| `comfy_nodes/ai_studio_h3_prompt/multi_shot.py` | the multi‑shot node — one long script → LLM shot plan → every shot rendered in order → one continuous video. `mode` = `fl2va` (text / chained first‑frame) or `ref2va` (every shot conditioned on the reference stills for a locked look). |
| `comfy_nodes/ai_studio_h3_prompt/script_tools.py` | the script‑writer + Gemini script‑review + **reference‑image generator** + Gemini video‑review nodes |
| `comfy_nodes/ai_studio_h3_prompt/_image_driver.py` | subprocess the `H3 Reference Images` node shells out to (under `chatgpt_python`) to drive Gemini / Flow image generation, one still per entity |
| `comfy_nodes/ai_studio_h3_prompt/system_prompts/fl2va.txt` | verbatim Claude rule file for text / first‑last‑frame ("three fields") |
| `comfy_nodes/ai_studio_h3_prompt/system_prompts/ref2va.txt` | verbatim Claude rule file for reference mode ("six sections") |
| `comfy_nodes/ai_studio_h3_prompt/system_prompts/script_writer.txt` · `story_expander.txt` · `director.txt` · `script_review.txt` · `video_review.txt` | editable instructions for the script / expander / director / QA nodes — no code change |
| `comfy_nodes/ai_studio_h3_prompt/system_prompts/camera_moves.md` · `film_extensions.md` | the two libraries the **advanced film‑making pass** folds into every shot — a camera‑move catalogue (canonical copy: `output/sample_prompts.md`) and cinematography building blocks + named feature‑film "house looks". Plain markdown, edit freely. |
| `workflows/AI Studio H3 - Prompt Writer + Text to Video (fl2va).json` | Ep29 text‑to‑video graph with the writer node in place of the prompt box |
| `workflows/AI Studio H3 - Prompt Writer + Reference Images (ref2va).json` | Ep29 three‑reference‑image graph, same swap |
| `workflows/AI Studio H3 - Multi-Shot Lengthy Video (fl2va).json` | lengthy‑video graph: split a script into shots and render them all, concatenated into one MP4 |
| `workflows/AI Studio H3 - Scripted Multi-Shot Lengthy Video (fl2va).json` | full chain: premise → script → Gemini review → render → Gemini video QA |
| `install_node.ps1` | junctions the node into the Pinokio ComfyUI `custom_nodes` **and** `workflows/` into its `user\default\workflows\AI Studio` |
| `comfy_nodes/ai_studio_h3_prompt/vendor/browser_llm.py` | bundled copy of the Playwright automation the `chatgpt` backend uses (no dependency on the AI Studio v2 checkout) |
| `comfy_nodes/ai_studio_h3_prompt/vendor/browser_profiles/chatgpt/` | bundled logged‑in Chromium profile for `chatgpt` — the whole reason this folder can be relocated (see **Relocating this folder** below) |

To change *how* prompts are written, edit the `.txt` files — no code change, no restart
of anything but ComfyUI. They are exact copies of
`…/H3 Prompt Formulas/Claude/H3 *(…)* - INSTRUCTIONS (paste).txt`.

---

## One‑time setup

1. **Link the node + workflows into ComfyUI** (already done once; re‑run if you move this folder):

   ```powershell
   & "<this folder>\install_node.ps1"
   # ComfyUI somewhere else?  add:  -ComfyApp "D:\path\to\ComfyUI\app"
   ```

   The junction *sources* are resolved from the script's own location, so a move
   just needs a re‑run. It makes two directory junctions into the ComfyUI app
   (default `…\minimax-h3-pinokio.git\app`): `comfy_nodes\ai_studio_h3_prompt` →
   `custom_nodes\`, and this folder's `workflows\` → `user\default\workflows\AI Studio`.
   The workflow JSONs then show up in the ComfyUI front end under
   **Workflows ▸ AI Studio**, and any edit you save there writes straight back
   into this folder's `workflows\`.

2. **Install Pixaroma nodes** (the workflows keep Ep29's size / duration / save‑mp4 helpers).
   In ComfyUI: **Manager → Custom Nodes Manager → search "Pixaroma" → install `ComfyUI-Pixaroma`**.

3. **Restart ComfyUI** — Pinokio → `minimax-h3-pinokio` → Stop, then Start — and hard‑refresh the browser tab.

4. **Have a backend ready** (see below).

5. Load a workflow: open the **Workflows** sidebar → **AI Studio** → pick one
   (or **Workflow → Open** and browse to `AI Studio ComfyUI\workflows\`).

---

## The node

**`H3 Prompt Writer (AI Studio)`** — category *AI Studio/H3*.

| Widget | Meaning |
|---|---|
| `brief` | plain‑language description of the video. This is the only field you normally touch. |
| `mode` | `fl2va` (text / first‑last frame — three fields) or `ref2va` (reference images/audio — six sections). Pre‑set per workflow. |
| `backend` | `chatgpt` (default) or `ollama`. |
| `duration_seconds` | `0` = let the writer decide / ask. Set it for FL2VA / multi‑shot so the writer doesn't stop to ask. Match it to the Pixaroma duration node. |
| `extra_instructions` | appended to your brief — e.g. `9:16 vertical, hard cuts only, no on‑screen text`. |
| `ollama_model` / `ollama_url` | for the ollama backend. Default `qwen3:32b` at `http://127.0.0.1:11434`. |
| `chatgpt_python` | interpreter that has Playwright (see below). Default is the machine's Python 3.14; point it anywhere `pip install playwright` lives. |
| `v2_dir` | folder with `browser_llm.py` + `browser_profiles/`. **Leave blank** to use the copy bundled in `vendor/`. Only set it to fall back to the AI Studio v2 checkout. |
| `timeout_s` | per‑call ceiling. |
| `regen` | bump by 1 to force a fresh call — the node caches on identical inputs, so re‑queuing alone won't re‑ask. |
| `system_prompt_override` | paste a whole rule file here to override `system_prompts/<mode>.txt` for this node only. |

Outputs `h3_prompt` (into the H3 node) and `info`. After it runs, the finished prompt
is shown read‑only on the node. If the writer needs a duration or a missing detail it
returns a **one‑line question** instead of a prompt — the `info` line is prefixed
`QUESTION`; answer it in `brief`, or set `duration_seconds`, and run again.

### Backend: `chatgpt` (default)

Drives your signed‑in **ChatGPT web** session through the bundled
`vendor/browser_llm.py` (Playwright, no API key) — the rule file and your brief go in
as one first message, exactly like the "send the whole file as your first message"
setup in the Claude *HOW TO USE*.

- It runs under `chatgpt_python` (default
  `C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe`), **not** the
  ComfyUI python, because it needs Playwright + a logged‑in Chromium profile. That
  profile is bundled at `vendor/browser_profiles/chatgpt/`.
- A Chromium window opens on each run (ChatGPT blocks headless). Don't close it.
- First time / if it times out logging in (run against the bundled copy):
  ```powershell
  $py  = "C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe"
  $dir = "<this folder>\comfy_nodes\ai_studio_h3_prompt\vendor"
  $env:BROWSER_PROFILES_DIR = "$dir\browser_profiles"
  & $py "$dir\browser_llm.py" chatgpt-login
  ```

### Backend: `ollama`

Direct HTTP to a local Ollama server. Fully offline.

```powershell
ollama serve            # if not already running
ollama run qwen3:32b    # make sure the tag is pulled (first run downloads it)
```

If the configured model isn't installed the node fails with the list of tags Ollama
actually has, so you can correct `ollama_model`.

---

## Using a workflow

1. Open `AI Studio H3 - Prompt Writer + Text to Video (fl2va).json`.
2. `1. Pick Orientation and Size` (Pixaroma) → orientation + resolution.
3. `2. Select Duration` (Pixaroma) → clip length. Put the **same** number in the writer's
   `duration_seconds` if you want timed multi‑shot cuts.
4. **`Write the H3 prompt`** node → type your description in `brief`. Optionally add
   `extra_instructions`.
5. Queue. The writer runs first (ChatGPT window opens, or Ollama churns), its output
   feeds the H3 node, then KSampler renders. Saved MP4 lands in
   `…/minimax-h3-pinokio.git/app/output/`.

For **ref2va**: also load your three reference images into the `3./4./5. Load Image`
nodes (drop files into `…/app/input/` or use the node's picker), and set `mode` is
already `ref2va`.

### Hand‑writing a prompt instead

Disconnect the writer node (or delete it) and type directly into the H3 node's `prompt`
field. The rest of the graph is unchanged from Ep29.

---

## Lengthy videos: one script → many shots

`AI Studio H3 - Multi-Shot Lengthy Video (fl2va).json` + the **`H3 Multi-Shot Lengthy
Video (AI Studio)`** node turn a whole narrative into one long clip.

**Flow:** paste the full script into `script` → the node makes **one** LLM call
(same H3 fl2va rules + a multi‑shot wrapper) and gets back an ordered **shot plan**:
each shot is a complete standalone H3 prompt plus a `continuation` flag. Then the node
renders the shots **one by one inside itself** (reusing the stock
`MiniMaxH3ImageToVideo → ConditioningZeroOut → KSampler → VAEDecode/VAEDecodeAudio`
pipeline) and concatenates every shot's frames + audio into a single `IMAGE`/`AUDIO`
pair wired to `PixaromaSaveMp4`. No ffmpeg, no loop‑node pack — one `output/Lengthy_*.mp4`.

`continuation` per shot:

| value | meaning |
|---|---|
| `false` | a fresh cut — new angle / location / time. Rendered from text only. |
| `true` | the same unbroken take — rendered **starting from the previous shot's last frame** (fl2va `first_frame`), so it carries straight on. Shot 1 is always `false`. |

The planner mixes the two per the script, so a scene that flows continuously gets
chained shots and a scene that jumps gets clean cuts.

| Widget | Meaning |
|---|---|
| `script` | the whole narrative. The only field you normally touch. |
| `width` / `height` / `frames_per_shot` | wired from the Pixaroma size / duration nodes. **Total length = `frames_per_shot` × number of shots.** |
| `seconds_per_shot` | advisory pacing target handed to the planner. |
| `max_shots` / `min_shots` | hard cap / lower bound on the plan (plan is truncated to `max_shots`). |
| `backend` | `chatgpt` (Chromium window opens once for the plan) or `ollama` (fully local). |
| `seed` | `0` = random base seed each run; set a value for a repeatable render (shot N uses `base + N`). |
| `steps` / `cfg` / `sampler_name` / `scheduler` / `denoise` | the KSampler settings that used to live on the graph. Ep29 defaults: 20 / 1.0 / `res_multistep` / `simple` / 1.0. |
| `plan_override` | paste a JSON array of shots to re‑render a hand‑tuned plan with **no** LLM call. |
| `regen` | bump to force a fresh plan + render. |
| `system_prompt_override` | replace the fl2va rule file for the planner. |
| `hud_timestamp` | CCTV‑style clock burned onto **every** frame in post (PIL, e.g. `01:42 AM`). Blank = off. Guarantees the timestamp is pixel‑identical in every shot instead of drifting. |
| `hud_position` | `top-right` · `top-left` · `bottom-right` · `bottom-left`. |
| `hud_tick` | off = the exact string on every frame; on = the seconds advance across the video (live‑feed look). |
| `cta_text` | engagement caption composited over the **first `cta_fraction`** of the video (uppercased, translucent pill, fades in/out), e.g. `Hit the follow button, heart, comment and share this video`. Blank = off. |
| `cta_fraction` | portion of the video from the start that shows the CTA (default `0.5` = first half). |
| `cta_position` | `bottom` · `top` · `center`. |
| `speed_lora` | filename of a turbo / few‑step LoRA under `models/loras` (e.g. `h3\minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors`, lightx2v *Minimax‑h3‑Turbo*). Blank = full‑quality. When set: `steps`→8 and `sampler_name`→`euler` **unless you changed them from the factory default**, and `sigma_shift` defaults to 6. ~⅓ the render time. |
| `speed_lora_strength` | LoRA weight, default `1.0`. |
| `sigma_shift` | H3 video flow shift. `0` = model default (12). The turbo LoRA is distilled for **6**; set with `speed_lora` and left at 0 → 6 is applied via `MiniMaxH3SigmaShift`. |
| `sigma_shift_audio` | audio flow shift paired with `sigma_shift` (H3 default `3`, keep it). |

**Turbo path (Pixaroma Ep32):** `speed_lora` + `sigma_shift 6` swaps in a distilled
few‑step LoRA and the schedule it was trained for. Steps stay at **8, not 4** — 4 is
enough for the picture but the *generated* audio is measurably worse (blind test
12 > 8 > 6 > 4). `scheduler = ddim_uniform` is **refused**: on every H3 model it discards
the shot you asked for (Pixaroma Ep32: 1.5/10, replaces the scene). Use `simple`,
`sgm_uniform` or `beta`. The `info` string shows a `turbo:` line with what actually ran.

The HUD / CTA are burned in **after** rendering, so they never wander between shots;
when either is set the planner is also told to keep every shot free of on‑screen
text / clocks / watermarks. If the overlay pass fails it is caught and noted in
`info` as `overlays FAILED (…)` — the render still saves.

**9:16 / vertical:** put `9:16 vertical` (or similar) in `extra_instructions`, or
just set a portrait `width`/`height`, and a **strict rule** is added to the planner
prompt — no cramped, boxed‑in rooms; every interior gets a high ceiling, a far
wall / doorway metres back, headroom and real depth (or an open location). `H3
Reference Images` location stills carry the same "open, airy, never cramped" rule.

Outputs `frames`, `audio`, and an `info` string (shot count, which shots were chained,
total frames/seconds, base seed, the first line of each shot's prompt, and an
`overlays:` line when the HUD / CTA pass ran).

**Cost / limits:** every shot's decoded frames are held in RAM until the final concat,
so a long render is memory‑hungry (≈ `frames × H × W × 3 × 4` bytes total). Keep
`max_shots` sane. Cancelling ComfyUI stops it between shots.

### Reference images — lock characters & locations across shots

`fl2va` from text alone drifts shot‑to‑shot (uniform colour, monitor count, a
burned‑in timestamp all wandering). The renderer now generates reference stills
itself and anchors every shot to them.

| Widget | Meaning |
|---|---|
| `auto_refs` | **on by default.** Before rendering, generate one clean still per element in `ref_entities` (or the script's `[CHARACTER:]` / `[ENVIRONMENT:]` / `[PROP:]` blocks) via `ref_backend`, save to `image_out_dir`, and feed them to every shot. Skipped when `ref_images` is wired in. |
| `ref_backend` | `gemini` (one persistent chat — the whole run shares context, so the look stays consistent) or `flow`. |
| `ref_entities` | one `kind \| name \| description` per line (`kind` = character / environment / prop / entity). Blank = read the script's entity blocks. |
| `shot_keyframes` | also generate a **start** and an **end** still for every shot and render it as a first↔last‑frame interpolation via `MiniMaxH3ImageToVideo`. A `continuation: true` shot skips its start still and reuses the previous shot's final rendered frame. Slower (2 image gens / shot). Implied by `mode = ref2va`. |
| `image_out_dir` | folder the stills + keyframes are written to (e.g. the story's output folder). Blank = a temp dir. |
| `image_timeout_s` | ceiling for the whole image‑generation subprocess (default 1200). |
| `mode` | `auto` · `fl2va` (+ refs as anchors) · `ref2va` (forces `shot_keyframes`). |
| `ref_images` / `ref_map` | wire from `H3 Reference Images` to supply stills yourself instead of `auto_refs`. |
| `ref_image_size` | `match` (scale refs to the render's pixel area — fast) or `max` (2048 px short edge — sharper, several × slower). |

**fl2va + refs** (default): each shot rendered through `MiniMaxH3ReferenceToVideo`
with all stills attached — same faces / set / props every cut, `continuation` forced
off, length snapped into the H3 reference range (`n % 17 == 5`, ≥ 124). **ref2va
keyframes**: adds the per‑shot start/end interpolation on top.

If the script fed in is **already broken into `[SHOT N]` blocks** (e.g. from the
`H3 Script Writer` / `H3 Script Review` nodes), the planner keeps those exact shot
boundaries and per‑shot `Duration`s instead of re‑cutting — one plan entry per
script shot.

### Advanced film‑making prompts — combine three sources

**On by default.** After the planner (ChatGPT / the `H3 Director` node) drafts each
shot, it runs one more pass that folds **three sources** into a single H3 prompt,
aiming for Hollywood‑feature‑grade shots — no per‑video prompt surgery, it is all
in the node's system prompt:

1. **the drafted shot** — its story, blocking, dialogue and beats (the spine);
2. **the camera‑move library** — `output/sample_prompts.md` (canonical, editable),
   falling back to the bundled `system_prompts/camera_moves.md`. The planner picks
   the one move a shot needs and writes its geometry + end framing as one
   in‑sentence clause, translating "smooth / gradual / quick" into H3's four
   allowed amplitude/speed phrases;
3. **the film‑look extensions** — `system_prompts/film_extensions.md`: format &
   capture, lens language, lighting, colour & grade, atmosphere, composition,
   score & sound, plus named feature‑film **house looks** (*Game of Thrones · The
   Matrix · TRON: Legacy · Avatar · The Odyssey · Interstellar · John Wick*). One
   look is chosen and held across the whole video.

| Widget | Meaning |
|---|---|
| `advanced_prompts` | on = run the three‑source enrichment pass. Off = plain planner output. |
| `film_look` | `auto` (planner picks from the script's `[GLOBAL LOOK]` / the material) · `none` (general cinematography grammar only, no named look) · a house‑look name (forced, held every shot). |
| `camera_moves_file` | override path to the camera‑move library (abs or relative to `v2_dir`). Blank = `output/sample_prompts.md` → bundled `camera_moves.md`. |
| `film_extensions_file` | override path to the film‑look library. Blank = `output/film_extensions.md` → bundled `system_prompts/film_extensions.md`. |

The libraries are plain markdown — edit them and the planner picks the change up on
the next run. The `H3 Director – Nolan` node has its own `film_look` widget: it
names a **`Reference look:`** in the `[GLOBAL LOOK]` header and holds every shot's
camera / lighting / palette / score to it.

Character reference stills now also lock the six things a character sheet locks
(concept, look, outfit, palette, style, pose) plus "don't restyle the hair / add
clothing / change proportions", so later shots have less to drift on.

---

## Scripted lengthy videos: premise → script → review → render → QA

`AI Studio H3 - Scripted Multi-Shot Lengthy Video (fl2va).json` chains four AI Studio
nodes around the multi‑shot renderer. All three new nodes keep their instructions in
`system_prompts/*.txt` (edit, no code change) and reuse the same browser / Ollama
plumbing as the prompt writer.

| Node | What it does |
|---|---|
| **`H3 Script Writer (AI Studio)`** | `premise` + `shots` + `seconds_per_shot` + `genre` → a shot‑by‑shot script (`[GLOBAL STYLE]` / `[GLOBAL ENTITIES]` / `[SHOT N]`) written to open on a hook and escalate. Backend `chatgpt` (default) or `ollama`. |
| **`H3 Story Expander (AI Studio)`** | a **short** story / premise / small shot list → a **longer, more detailed** script in the same `[SHOT N]` house format, running about `target_minutes` (default **2.5**). Shot count = `target_minutes`·60 / `seconds_per_shot` (capped 40). Keeps the source's premise, tone and **ending**; adds beats, locations, props, sensory texture and one or two payoff details. `extra_directions` folds in new elements. Instructions in `system_prompts/story_expander.txt`. Backend `chatgpt` (default) or `ollama`. Wire `story` → `H3 Script Review` or straight into the renderer's `script` (bump its `max_shots`). |
| **`H3 Director – Nolan (AI Studio)`** | a script → a shot-by-shot **director's breakdown** in the voice of Christopher Nolan. Per `[SHOT N]`: `Duration`/`Pace`, `Location` (geometry + real light sources), `Characters` (who, where, facing, posture, hands, state — flagged if only on a monitor), `Props & positions` (chair/desk/monitors/mug/knife/watch… where each sits + its state; a weapon stays hidden till used), `Camera` (one continuous setup — size, lens mm, height/angle, movement, start→end) + optional `Alt camera`, `Lighting` (key/fill/practicals/what falls to black), `Blocking / performance` (beat by beat), `Dialogue` (verbatim + delivery), `Music` (non-diegetic score), `SFX` (layered diegetic). Emits a `[GLOBAL LOOK]` header first. `keep_shot_count` on = keep the input's `[SHOT N]` boundaries 1:1. `look` steers the visual grammar; `film_look` anchors the `[GLOBAL LOOK]` to a named feature‑film "house look" (`auto` = the director picks; catalogue in `system_prompts/film_extensions.md`); `extra_directions` folds in staging notes. Instructions in `system_prompts/director.txt`. Backend `chatgpt` (default) or `ollama`. Wire `shot_breakdown` → the renderer's `script` (or → `H3 Script Review` first). |
| **`H3 Script Review – Gemini (AI Studio)`** | sends the script to **Gemini** to check logical + physical consistency — *horror / supernatural / surreal / fantasy may break physics on purpose* — and **restructures it**: `[SHOT 1]` becomes the strongest hook (cold open), then 1–2 context shots, then the story in order. Outputs `revised_script` (→ the renderer) and `review_notes`. Backend `gemini` (default) or `ollama`. |
| **`H3 Reference Images (AI Studio)`** | parses `[GLOBAL ENTITIES]` from the script and generates **one clean reference still per recurring character / location** (optionally props) via **Gemini** (Nano Banana) or **Flow**. Outputs `ref_images` (IMAGE batch) + `ref_map` (the `<Picture N> = …` index) → wire both into the renderer's `ref_images` / `ref_map` and set its `mode = ref2va`. `max_refs`, `include_props`, `entities_override`, `style_suffix`, `regen`. |
| **`H3 Multi-Shot Lengthy Video (AI Studio)`** | as above — `revised_script` is wired into its `script` input; it keeps the script's `[SHOT N]` blocks and renders them. With `ref_images` wired and `mode = ref2va`, every shot is conditioned on the reference stills (see **`mode = ref2va`** above). |
| **`H3 Video Review – Gemini (AI Studio)`** | samples `num_probe_frames` (default 12) evenly across the rendered `frames` and uploads them to **Gemini** with the script, asking whether the video **adheres to the script** and is logically / physically sensible (same genre exception). Set `video_path` to a real `.mp4` to upload the file instead of frames. Outputs `report` + `verdict` (`PASS` / `PASS WITH NOTES` / `FAIL`). |

`genre`: set it on the writer (`horror`, `sci-fi`, …); leave the two review nodes on
`auto` to let Gemini infer it and apply the physics exception itself.

**Backends / logins:** the `chatgpt` stages use `vendor/browser_profiles/chatgpt`; the
`gemini` stages use `vendor/browser_profiles/gemini`. Both run under `chatgpt_python`
(any interpreter with Playwright). One‑time Gemini login:

```powershell
$py  = "C:\Users\Admin\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$dir = "<this folder>\comfy_nodes\ai_studio_h3_prompt\vendor"
$env:BROWSER_PROFILES_DIR = "$dir\browser_profiles"
& $py "$dir\browser_llm.py" gemini-login
```

Each stage is cached on identical inputs — bump the node's `regen` to force a re‑run.

---

## Models the graphs expect (Ep29, unchanged)

`fl2va`: `diffusion_models/h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors` ·
`ref2va`: `…/minimax_h3_ref2va_pruned_int8_convrot.safetensors` ·
`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` ·
`vae/minimax_h3_video_vae_fp16.safetensors` + `vae/minimax_h3_audio_vae_fp32.safetensors`.
The workflow notes carry the download links.

---

## Relocating this folder

This folder is self‑contained. To move it (e.g. `E:\Work\AI Studio ComfyUI`
→ `D:\Studio\ComfyUI`):

1. **Move the whole folder** as‑is. Everything the node needs travels with it:
   the node code, both workflow JSONs, the rule files, and `vendor/`
   (`browser_llm.py` + the logged‑in `browser_profiles/chatgpt/`).
2. **Re‑run the linker** from the new location so the junctions point at it:
   ```powershell
   & "D:\Studio\ComfyUI\install_node.ps1"          # -ComfyApp "..." if ComfyUI moved too
   ```
3. **Restart ComfyUI** and hard‑refresh the tab.

Nothing in the folder hard‑codes `E:\Work\AI Studio ComfyUI` any more — `v2_dir` is blank
(resolves to `vendor/`), and the workflow JSONs ship it blank too.

**What is NOT in this folder** (must exist on whatever machine runs it):

| Need | For | Notes |
|---|---|---|
| A ComfyUI install with the MiniMax H3 **core** nodes + **ComfyUI‑Pixaroma** + the model files | the actual render | default target is the Pinokio `minimax-h3-pinokio` app; pass `-ComfyApp` to point elsewhere |
| Python with **Playwright** (`pip install playwright` + `playwright install chromium`) | the `chatgpt` backend only | set the node's `chatgpt_python` widget to it; the `ollama` backend needs none of this |
| A local **Ollama** server | the `ollama` backend only | not needed if you only use `chatgpt` |

**Different Windows user or different machine:** the bundled Chromium profile's
cookies are encrypted with Windows DPAPI tied to the *user account*, so ChatGPT
will be logged out there. Re‑run the `chatgpt-login` block from **Backend: `chatgpt`**
once to re‑seed `vendor/browser_profiles/chatgpt/`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Node missing after restart | `install_node.ps1` again; confirm the junction at `…/minimax-h3-pinokio.git/app/custom_nodes/ai_studio_h3_prompt`; check the ComfyUI log for an import error. |
| `browser_llm.py not found under v2_dir` | the `vendor/` copy is missing — restore `vendor/browser_llm.py`, or set `v2_dir` to an AI Studio v2 checkout. |
| `chatgpt_python not found` | set `chatgpt_python` to an interpreter that has Playwright (`pip install playwright && playwright install chromium`). |
| ChatGPT opens but isn't logged in / run "exceeded …s" | re‑run the `chatgpt-login` block from **Backend: `chatgpt`** (it must target `vendor/browser_profiles`); raise `timeout_s`. |
| `Cannot reach Ollama` | `ollama serve`. |
| Writer returns a question every run | set `duration_seconds`, or add the missing detail to `brief`, then bump `regen`. |
| Prompt has ```` ``` ```` fences or stray prose | the node already strips a single fenced block; if ChatGPT wrapped it oddly, copy from the read‑only box and clean up, or switch `backend` to `ollama`. |
