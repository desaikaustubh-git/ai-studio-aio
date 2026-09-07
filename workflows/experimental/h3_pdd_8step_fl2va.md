# H3 PDD 8-Step (Alibaba PAI Acc adapter) — single-pass FL2VA/T2V + audio

**New / experimental.** Does not touch any existing workflow. Files:

- `h3_pdd_8step_fl2va.json` — the graph (ComfyUI API format)
- this note

## What it is

MiniMax H3 accelerated with **Alibaba PAI's PDD (Phased Distribution
Distillation) 8-step "Acc" adapter** instead of the lightx2v turbo LoRA. PDD's 32
trained time intervals are fused 4-at-a-time into **8 joint audio+video model
forwards** — it is a purpose-trained distillation schedule, not a naïve 8-step
run over a 4-step LoRA.

The `MiniMaxH3PDD8StepSetupT8Advanced` node (from the T8 pack) takes the base
`MODEL` + the AV latent and emits **`MODEL` + `SAMPLER` + `SIGMAS` as one bundle**
on a fixed contract: `euler / simple / 8 NFE / video shift 12 / audio shift 3 /
CFG 1`, adapter strength `1.0`. So this graph has **no `MiniMaxH3SigmaShift`, no
`KSamplerSelect`, no `BasicScheduler`** — the PDD node owns all of that — and
**no other LoRA may be stacked** (an ordinary LoRA node drops PDD's dynamic
video/audio output heads).

This is the "proper PDD-aware split" that
`../h3_allinone_2stage_latentupscale.md` names as the real fix for that path's
checkerboard corruption (which came from `BasicScheduler(8) → SplitSigmas(4)`
violating the turbo LoRA's distilled 4-step schedule).

## Before it will run — three installs

1. **Enable the T8 node pack.** Rename
   `custom_nodes/comfyui-minimax-h3-audio-T8.disabled` →
   `custom_nodes/comfyui-minimax-h3-audio-T8` and restart ComfyUI. It was parked
   as `.disabled` on 2026-09-01 as a large, less-vetted ~200-file pack (face
   refine, lip-sync, a bundled LLM, skin pipelines, …) — see
   `../h3_allinone_2stage_latentupscale.md`. Only `MiniMaxH3PDD8StepSetupT8Advanced`
   is needed here; the rest can stay unused.

2. **Download the PDD adapter** into `models/loras/`:
   `MiniMax-H3-FL2VA-Acc-8Step_comfyui_pdd.safetensors`
   (Ref2VA build: `MiniMax-H3-Ref2VA-Acc-8Step_comfyui_pdd.safetensors`).
   Source: Alibaba PAI / the T8 pack's model list.

3. **Download the FULL non-pruned base** into `models/diffusion_models/`:
   `minimax_h3_fl2va_int8_convrot.safetensors`.
   The pruned file this box has (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`)
   will **not** satisfy the adapter — PDD checks for the 258-block backbone, the
   4 dynamic output heads and AdaLN width 2688 of the full build.

## Graph (node ids in the json)

```
1 UNETLoader (FULL fl2va base) ─┐
2 CLIPLoader ─┐                 │
3 VAELoader (video) ─┼─► 20 MiniMaxH3ImageToVideo ─┬ positive ─────────► 21 BasicGuider ─┐
4 VAELoader (audio) ─┘   768×1344, length 124      └ LATENT ─┬─► 8.av_latent            │
11 LoadImage ─ 12 ImageResizeKJv2(768×1344) ─► 20.first_frame │                          │
                                                              │  1.MODEL ─► 8.model      │
                              8 MiniMaxH3PDD8StepSetupT8Advanced ├─► MODEL ──────────────► 21
                              (adapter, "FL2VA", 1.0)           ├─► SAMPLER ─┐            │
                                                                └─► SIGMAS ─┐│            │
23 RandomNoise(seed) ─────────────────────────────────────────────────────┐ ││           │
30 SamplerCustomAdvanced(noise 23, guider 21, sampler 8.SAMPLER, sigmas 8.SIGMAS, latent 20.LATENT)
   ├─► 61 VAEDecode(video vae) ─┐
   └─► 62 VAEDecodeAudio(audio vae) ─┴─► 70 PixaromaSaveMp4 (fps 24, trim_to_audio)
```

`8.report_json` (STRING) is left unconnected — add a `ShowText`/preview node to
read PDD's validation report if you want it.

## FL2V vs T2V vs first+last

- **FL2V** (as wired): pick the image in node `11`; it's cropped to 768×1344 in
  `12` and fed to `20.first_frame`.
- **Pure T2V**: delete the link into `20.first_frame` (or Bypass nodes `11`+`12`).
- **First + last**: add a second `LoadImage`→`ImageResizeKJv2` into `20.last_frame`.

## Knobs

| node | field | ships as | notes |
|---|---|---|---|
| 20 | `prompt` | rain-window portrait demo | canonical 3-field style, `00:0X` markers |
| 20 + 12 | `width`/`height` | 768×1344 | keep the two matched; start small (736×416) for the first serial test on 16 GB |
| 20 | `length` | 124 | 17k+5 grid |
| 8 | `variant` widget | `FL2VA` | must match the adapter file and the base; `Ref2VA` is not interchangeable |
| 8 | `strength` widget | `1.0` | official — don't change without a separately trained contract |
| 23 | `noise_seed` | 0 | |

Steps / scheduler / both sigma shifts are **locked by the PDD node** — do not add
nodes to override them.

## Where it fits vs. the CCTV v22 pipeline

v22 uses `AIStudioH3MultiShotRender`'s core recipe: turbo LoRA @1.0 + realism
LoRA @0.9, 8 steps euler/simple, sigma_shift 6/3. PDD is the **same 8-step
budget** but via a trained distillation schedule. Expect a different motion / skin
character — A/B one CCTV shot against its v22 render before committing.
