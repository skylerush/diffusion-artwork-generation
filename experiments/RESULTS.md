# Experiments log — Diffusion Models for Impressionist Artwork

Every run records: config + seed + final metrics + sample artifacts + a short note
(**Hypothesis / Result / Decision**). This table is the spine of the "show the process"
deliverable (failed runs, analysis, hyper-parameter search). Fill metrics as runs finish.

Hardware: RTX 5090 (32 GB, sm_120), torch 2.11.0+cu128, Python 3.12.

---

## Phase 1 — From-scratch DDPM

| Run | Dataset | Key setting | Steps | Final loss | Verdict |
|-----|---------|-------------|-------|-----------|---------|
| `p1a_butterflies` | butterflies-64 | base128, cosine, lr2e-4, EMA+warmup | 8000 | ~0.011 | ✅ clear butterflies; EMA bug found→fixed |
| `p1-r01` *(failure ladder)* | butterflies-64 | **lr 2e-3** (10× high) | 2000 | ~0.98 | ✅ stalls — overshoots, never converges; confirms lr 2e-4 |
| EMA bug *(real failure→fix)* | butterflies-64 | decay 0.9999, no warmup | — | — | ✅ EMA=noise @4k (≈67% init) → warmup + `--reset-ema` → recovered (supersedes planned synthetic "no-EMA") |
| `p1b_impressionism` | impressionism-64 | from scratch @64px | 8000 | ~0.04 | ✅ learns palette/texture, not coherent scenes — too-diverse data at 64px; motivates pretrained prior (vs coherent butterflies) |

**Notes log:**
- _p1a_butterflies_ — **Hypothesis:** the standard DDPM recipe yields recognizable butterflies by ~10–15k steps.
  **Result:** the *raw* model produced clear butterflies already by **step 4000** (faster than expected) — but the
  **EMA** samples were pure noise. **Root cause:** EMA decay 0.9999 averages ~1/(1−decay)=10k steps, so at step 4k
  the shadow was 0.9999⁴⁰⁰⁰≈**67% random initialization** → noise. **Decision:** added EMA **warmup**
  `(1+t)/(10+t)→0.9999` and resumed with `--reset-ema` (re-seed shadow from the good step-4000 weights) — no
  training lost. Now logging both raw + EMA samples to watch EMA recover. *(A real failure→fix, replacing the
  planned synthetic "no-EMA" one.)*

---

## Phase 2 — Stable Diffusion fine-tuning (Impressionism)

Base: `stable-diffusion-v1-5/stable-diffusion-v1-5` @ 512px, bf16.

| Run | Method | Key setting | Steps | FID↓ | CLIP↑ | Trainable | Verdict |
|-----|--------|-------------|-------|------|-------|-----------|---------|
| `base` | none | SD-1.5 baseline | — | 167.4 | 33.48 | 0 | baseline (already cues "impressionist") |
| `lora_r4` | LoRA | rank 4 | 1500 | **159.2** | 33.37 | ~0.8M | ✅ strong; nominal-lowest FID (≈noise) |
| `lora_r16` | LoRA | rank 16 | 1500 | 164.1 | 33.15 | ~3.2M | ✅ strong visual quality; FID mid-band |
| `lora_r64` | LoRA | rank 64 | 1500 | 166.0 | 32.84 | ~12.8M | drifts (re-composes scene); no gain |
| `full_ft` | Full FT | lr 1e-6, grad-ckpt | 1500 | 164.1 | **33.49** | 860M | matches LoRA at ~270× the cost |
| `dreambooth` | DreamBooth | +prior preservation | 1200 | 162.9 | 32.47 | ~3.2M | good FID; softens detail, lowest CLIP |

**Takeaway (as written at the time):** all methods cluster (159–167 FID; base SD already strong) —
small LoRA matches full fine-tuning; FID at N=256 is noisy, so the same-seed visual grid + CLIP carry
more signal than FID.

> ### ⛔ THE v1 TABLE ABOVE IS RETRACTED — DO NOT CITE THESE NUMBERS
> They used **N = 256 generated vs 300 real**. FID fits a **2048-dimensional** Gaussian, so that
> covariance is rank-deficient (rank ≤ 255 of 2048) and the score is dominated by estimator bias.
> Proof: **two sets of genuine Impressionist paintings scored FID 156.7 against each other** — every
> model above sits within ~10 points of a floor that a *perfect* generator could not beat.
> **v1 measured the estimator, not the models.** Three of its conclusions are overturned below
> (rank-64, "r4 best", and DreamBooth's untested trigger). Kept visible on purpose: **the retraction
> is itself a result.** See `JOURNEY.md` §8.

---

## Phase 2 — Evaluation **v2** (final, corrected ruler)

2,048 imgs/model vs **2,800** held-out refs · **neutral prompts (zero style words)** · floor re-measured.
**Real-vs-real floor: FID 37.6** (was 156.7 at v1's N — pure estimator bias). Source: `eval_v2_results.json`.

| Model | FID ↓ | above floor | CLIP ↑ | Verdict |
|---|---|---|---|---|
| base | 128.3 | +90.7 | 32.72 | beaten by every fine-tune |
| lora_r4 @×1.5 | 116.9 | +79.3 | 32.67 | 3.3 MB; beats both full-FTs |
| lora_r16 @×1.0 | 119.3 | +81.7 | 32.89 | scale-1.0 control |
| **lora_r16 @×1.5** | **112.8** | **+75.2** | **32.93** | 🏆 **best model overall** |
| lora_r64 @×1.5 | 114.5 | +76.9 | **33.10** | 2nd place, highest CLIP — v1 "worst" verdict revised |
| dreambooth @×1.5, no trigger | 119.65 | +82.1 | 32.71 | beats both full-FTs **without** its `sks` token |
| dreambooth @×1.5, **+ `", in sks style"`** | 119.66 | +82.1 | **31.66** | 🆕 trigger adds **no** style, costs **−1.05 CLIP** |
| full_ft (4 img/step) | 123.0 | +85.4 | 32.78 | v1 confounded run |
| full_ft_matched (8 img/step) | 121.5 | +83.9 | 32.84 | fair fight — still loses to LoRA |

**v2 verdicts:** fine-tuning beats base by up to **−15.5 FID** with no prompt hint · **LoRA beats matched
full-FT by −8.7** (headline upgrades *matches → beats*) · **every adapter (3.3–51 MB) beats every full
fine-tune (3.44 GB)** · scale ×1.0→×1.5 = **−6.5 FID free** · r16 > r4 at matched scale (v1's "r4 best"
= noise) · images-seen confound closed (matched helped only +1.5).

**Two v1 conclusions overturned, not merely sharpened:**

- **🔄 "Rank saturates; r64 drifts for zero gain" — RETRACTED.** r64 is **2nd overall (114.5)** with the
  **highest CLIP of all nine (33.10)**. It beats r4 by 2.4 FID, so "16× the params for zero measured
  gain" is false. What survives is a **shallow optimum near r16** (r4 116.9 → r16 112.8 → r64 114.5),
  and rank tracking prompt adherence (32.67 → 32.93 → 33.10).
- **🆕 "DreamBooth trades flexibility for token control" — the token does nothing.** FID is identical
  to **0.01** with and without `sks`; only CLIP moves, and *downward*. The style bound to the weights
  **unconditionally**, never to the token — 1,200 instance images under one fixed prompt give `sks`
  no contrastive signal to isolate. **v1 never tested this**: it scored DreamBooth only on prompts
  that omit the token, i.e. with the method's mechanism switched off.

**Honest caveats.** +75 above floor = still clearly distinguishable from real Impressionism (expected at
1,200 training images) · v1 and v2 numbers are **not comparable** (different prompts *and* N) · `base`
and `lora_r4` were generated at batch 16 vs batch 8 for the rest (the D10 VRAM fix); all other settings
identical and FID is distributional, so no bias is expected — but this is **assumed, not verified** ·
`report/figures/method_compare.png` still shows the **v1 regime** (style-word prompts at scale 1.0).

---

## Final method comparison (filled at the end)
LoRA (rank sweep) vs full fine-tune vs DreamBooth — across **quality (FID)**,
**text-alignment (CLIP score)**, **training cost** (trainable params / VRAM / wall-clock),
and **qualitative style adherence**. Reference for FID = held-out Impressionism set
(`data/impressionism_512/heldout`, never trained on).
