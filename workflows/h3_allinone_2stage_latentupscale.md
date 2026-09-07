# H3 All-in-One — single-pass (T2V / I2V / FL)

Started as a rebuild of the **Prompt Mastery** video *"All-in-One MiniMax H3
Workflow — Best for Action Shots (T2V + I2V, Latent Upscale)"*
(`youtube.com/watch?v=rOB7rLu3Y9w`) — its two-stage sigma-split + trained
latent-upscale pipeline. That version **ran successfully but produced
checkerboard/color-patch image corruption**, root-caused and reverted; see
**KNOWN ISSUE** below for the full story if you want to revisit it. This file is
now the reverted, verified-clean **single-pass** form.

Files:
- `h3_allinone_2stage_latentupscale.json` — the graph (API format)
- also copied to `…\minimax-h3-pinokio.git\app\user\default\workflows\` so it shows in the Workflow menu

## Load it

Drag the `.json` onto the ComfyUI canvas — 0.31.0 detects the API format and offers to
import (auto-arranges the nodes). Or **Workflow ▸ Open**.

## Graph (node ids in the json)

```
1 UNETLoader ─┐
5 Lora(turbo 1.0) ─ 6 Lora(realism 0.9) ─ 7 SageAttnPatch ─ 8 SigmaShift(6/3) ─┬─► 21 BasicGuider
2 CLIPLoader ─────────────────────────────────────────────┐                    ├─► 24 BasicScheduler(simple, 8 steps, full schedule - no split)
3 VAELoader (video) ─┐                                     │                    │
4 VAELoader (audio) ─┼──────────────────┐                  │
11 LoadImage ─ 12 ImageResizeKJv2(1344×768) ─ first_frame ─►20 MiniMaxH3ImageToVideo(1344×768, len 362) ─┬ positive ─►21
                                                                                                         └ LATENT ─────► 30

22 KSamplerSelect(euler)   23 RandomNoise(seed)

30 SamplerCustomAdvanced(noise, guider=21, euler, sigmas=24 FULL, latent_image=20.LATENT)
   ├─► 61 VAEDecode(video vae)  ──────┐
   └─► 62 VAEDecodeAudio(audio vae) ──┴─► 70 PixaromaSaveMp4(fps 24, trim_to_audio, video+audio)
```

Decodes straight off the raw sampled AV latent — no `LTXVSeparateAVLatent` /
`LTXVConcatAVLatent` needed without an upscale stage splitting video from audio
in between. This is the exact recipe every other H3 clip this session used
successfully (`AIStudioH3MultiShotRender`'s core path: 8 steps, euler, simple,
cfg 1.0 via turbo LoRA, sigma_shift 6/3).

## The 3 knobs

1. **`20.prompt`** — the shot. Written in `integrated_multimodal_description / overall_soundscape / non_diegetic_music` style with `00:0X` time markers. Ships with a 15 s rooftop-chase demo.
2. **`20.length`** (+ matching `12`/`20` `width`/`height`) — frames @ 24 fps, on the **17k+5 grid** (`362` = 15.08 s; `175` = 7.29 s; `124` = 5.17 s). Resolution ships at 1344×768 (16:9); any H3-friendly res works, just keep `12` and `20` matched.
3. **`24.steps`** — sampling steps. `8` matches every other render this session. Don't split this schedule across two passes with the turbo LoRA loaded (see KNOWN ISSUE).

## T2V vs I2V vs first/last-frame

- **I2V / FL** (as wired): pick the image in node `11`, it's cropped to 1344×768 in `12`, fed to `20.first_frame`.
- **Pure T2V**: delete the link into `20.first_frame` (or set nodes `11`+`12` to Bypass). Leave `example.png` selected so the graph stays green.
- **First + last frame**: add a second `LoadImage`→`ImageResizeKJv2` and wire it into `20.last_frame`.

## Substitutions vs the video (still true for the single-pass form)

| Video (Prompt Mastery)                         | This build                                             |
|------------------------------------------------|---------------------------------------------------------|
| Hybrid **FL2VA/REF2VA** checkpoint (layers 25–49 merged) | `minimax_h3_fl2va_pruned_int8_convrot` (the FL2VA half — you don't have the merge; `MiniMaxH3ImageToVideo` covers T2V + I2V + first/last frame) |
| 4-step **Turbo LoRA**                          | `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy` @ 1.0 |
| "oily-skin fix" LoRA                           | `h3-realism-people-t2v-i2v-r2v` @ 0.9                  |
| `ModelAttentionBackend`                        | `MiniMaxH3MemoryEfficientSageAttentionPatch`          |
| `VHS_VideoCombine`                             | `PixaromaSaveMp4`                                     |
| Two-stage sigma-split + trained latent upscale | **not used** — see KNOWN ISSUE                        |

## ⚠️ KNOWN ISSUE (why there's no upscale stage here): checkerboard corruption

The two-stage sigma-split + `MinimaxH3LatentUpscaler3D` version of this workflow
**ran without errors** (211s, valid 15.07s mp4, 1344×768) but the picture itself
was corrupted — a recognizable figure buried under dense colored-patch noise.

**Root-caused, not guessed:** decoded stage 1's latent directly, bypassing the
upscaler and stage 2 entirely (`h3_stage1_only_*.png` in the ComfyUI output
folder) — already pure checkerboard, no coherent image at all. So the trained
`MinimaxH3LatentUpscaler3D` node (installed from
`xmarre/Comfyui_Minimax_h3_latent_Upscaler`, weights in
`models/latent_upscale_models/`, confirmed registering and running with no
errors) was innocent. The corruption is in the very first sampling pass.

**Prime suspect:** `minimax_h3_fl2v_lightx2v_turbo_4step` is a LoRA distilled for
exactly 4 steps. The two-stage graph drove it with a generic
`BasicScheduler(8, "simple")` → `SplitSigmas(step 4)` — an 8-step schedule
mechanically cut into two 4-step halves, not the actual sigma schedule the
turbo LoRA was distilled for. That's a well-known way to get exactly this
checkerboard artifact with a distilled model. This is precisely the problem
`MiniMaxH3DualClockSamplerT8` exists to solve with a proper PDD-aware split —
see below for why that pack wasn't installed.

**If you want the two-stage/upscale version back:** the fully-wired graph (both
`MiniMaxH3ImageToVideo` stages, `SplitSigmas`, `MinimaxH3LatentUpscaler3D`,
correct DynamicCombo `mode`/`mode.width`/`mode.height` flat-key shape, the
stage-2 conditioning fix) is preserved in this conversation's history — ask and
I'll restore it — but expect the corruption until the sigma split itself is
fixed (options 2/3 below).

**Options if you want to revisit it:**
1. ~~Use this graph single-pass~~ — **done, this file now IS that.**
2. Get a genuine PDD-aware sigma split for the turbo LoRA instead of the naive
   `BasicScheduler`/`SplitSigmas` cut (hand-derive the trained schedule, or
   pull just that logic out of the T8 pack below without loading the rest of
   it) — not attempted.
3. Drop the turbo LoRA for the two-stage path and use full-step (20+) sampling,
   which has no distillation schedule to violate — much slower, loses the
   speed benefit that's the whole point of this workflow.

### Why the dual-clock sampler pack was NOT installed

`MiniMaxH3DualClockSamplerT8` lives inside `T8mars/comfyui-minimax-h3-audio-T8`
— not a small single-node add like the latent upscaler. That repo is a
sprawling ~200-file grab-bag: face/multiface refinement, lip-sync
(`mv_lipsync`), a bundled "prompt_rewriter_8b" LLM, skin-reconstruction
pipelines with "person_profiles", plus its own SLA/PDD LoRAs and acceleration
nodes — almost none of it related to sampling. That's a materially bigger and
less-vetted surface than the name suggested, so it was cloned, then immediately
renamed to `*.disabled` (ComfyUI skips `*.disabled` folders in `custom_nodes/`)
rather than loaded. It sits inert at
`custom_nodes/comfyui-minimax-h3-audio-T8.disabled` if you want to inspect it
yourself.

## Verification log (2026-09-01)

- Cloned `xmarre/Comfyui_Minimax_h3_latent_Upscaler`, downloaded
  `minimax_h3_latent_upscaler_3d_fp16.safetensors` (659 MB) into
  `models/latent_upscale_models/`.
- Found + disabled a stale duplicate ComfyUI process masking a failed restart;
  one clean relaunch got `MinimaxH3LatentUpscaler3D` registering in `/object_info`.
- Cloned then `.disabled`-renamed `T8mars/comfyui-minimax-h3-audio-T8` (see above).
- Two-stage graph: fixed a `mode` DynamicCombo shape bug (flat dotted keys, not
  nested) and a stage-2 conditioning shape-mismatch crash (needs its own
  `MiniMaxH3ImageToVideo`+`BasicGuider` at the target resolution — confirmed
  against the upscaler repo's own shipped example workflow).
- Two-stage graph then ran clean (211s) but output was corrupted; root-caused
  to the sigma split (above), not the upscaler.
- Reverted to this single-pass form; validated, rendered end to end (~3 min, 15.07 s, 1344×768, 12 MB) — frames clean, no corruption, matches the prompt.
