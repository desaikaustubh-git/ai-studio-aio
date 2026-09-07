# H3 FastVideo (FH3) — 4-step distill, single-pass T2V + audio

**New / experimental.** Does not touch any existing workflow. Files:

- `h3_fastvideo_fh3_t2v.json` — the graph (ComfyUI API format)
- this note

## What it is

MiniMax H3 run through the **FastVideo lab's distilled "FastH3" checkpoint**
(`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`), redistributed by
Kijai as an INT8 convrot safetensors. It's a fine-tune of base H3 that bakes in
**4-step step-distillation** and **VSA (Video Sparse Attention)** — so you drop
the lightx2v turbo LoRA entirely and sample the checkpoint directly. FastVideo's
own guidance is to run it at **8 steps** (4 works, 8 is cleaner); this graph
ships at 8.

Covered in *"MiniMax H3 Just Got Fast"* — Benji's AI Playground
(`youtube.com/watch?v=Y9J-dXJhz-M`).

## Before it will run

Place the checkpoint here:

```
models/diffusion_models/minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors
```

~22 GB, from **huggingface.co/Kijai/MiniMax-H3-experimental**. It is **not on this
box** — `UNETLoader` will error until the file is present. Everything else in the
graph (CLIP, both VAEs, the SageAttention patch, `MiniMaxH3SigmaShift`,
`MiniMaxH3ImageToVideo`, `PixaromaSaveMp4`) is already installed and is the same
set the proven single-pass recipe uses.

## Graph (node ids in the json)

```
1 UNETLoader (FastH3 ckpt) ─ 7 SageAttnPatch ─ 8 SigmaShift(12/3) ─┬─► 21 BasicGuider
2 CLIPLoader ─┐                                                     ├─► 24 BasicScheduler(simple, 8 steps)
3 VAELoader (video) ─┼─► 20 MiniMaxH3ImageToVideo (T2V: no first_frame) ─┬ positive ─► 21
4 VAELoader (audio) ─┘   768×1344, length 124                            └ LATENT ──► 30

22 KSamplerSelect(euler)   23 RandomNoise(seed)
30 SamplerCustomAdvanced(noise 23, guider 21, sampler 22, sigmas 24, latent 20.LATENT)
   ├─► 61 VAEDecode(video vae) ─┐
   └─► 62 VAEDecodeAudio(audio vae) ─┴─► 70 PixaromaSaveMp4 (fps 24, trim_to_audio)
```

Same topology as `../h3_allinone_2stage_latentupscale.json` **minus the two LoRA
nodes** and with `first_frame` left unconnected (pure text-to-video).

## Knobs

| node | field | ships as | notes |
|---|---|---|---|
| 20 | `prompt` | blue-hour rooftop demo | `integrated_multimodal_description / overall_soundscape / non_diegetic_music`, `00:0X` time markers |
| 20 + graph | `width`/`height` | 768×1344 (~1.03 MP) | drop to ~640×1152 to hit FastVideo's ~30 s / 5-s-clip figure; keep 32-aligned |
| 20 | `length` | 124 (5.17 s) | 17k+5 grid: 124 / 175 / 362 |
| 24 | `steps` | 8 | 4 = max speed, 8 = cleaner |
| 8 | `shift_video` | 12.0 | core default; sweep 6–12 if motion looks over/under-shifted. `shift_audio` 3.0 |
| 23 | `noise_seed` | 0 | |

Optional: to fight plasticky skin, add
`LoraLoaderModelOnly(h3-realism-people-t2v-i2v-r2v.safetensors @ 0.9)` between
node 1 and node 7. Untested with this distill — A/B it.

## Where it fits vs. the CCTV v22 pipeline

v22 is **ref2va** — four reference plates carry the guard / mannequin / booth /
CRT identity across shots. FH3 is **text-to-video only**: cross-shot consistency
has to come from the written prompt. Use FH3 for speed, concepting, or
prompt-only storytelling — not for the locked-character short.
