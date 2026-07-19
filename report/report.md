# Diffusion Models for Impressionist Artwork Generation
### Neural Networks — Final Project Report
**Sagie Zaoui · Amit Sachs** — Shenkar College

> Status: **complete.** Every metric here came from an actual logged run (`experiments/RESULTS.md`,
> `experiments/sweep.log`); figures are in `report/figures/` and `outputs/`.
>
> 📔 **The full process narrative — roadmap, timeline, and every failure → diagnosis → fix — lives in
> [`JOURNEY.md`](../JOURNEY.md)**, with a visual timeline at `report/figures/journey_timeline.png`.

---

## Abstract
We study diffusion models for generating Impressionist artwork in two stages. First, to build a
mechanistic understanding, we implement a Denoising Diffusion Probabilistic Model (DDPM) **from
scratch** in PyTorch — a U-Net noise predictor, cosine noise schedule, ε-prediction objective, and
EMA — and train it on 64×64 images; it produces recognizable samples and surfaces a real engineering
pitfall (an EMA whose decay was too high for the run length, which we diagnose and fix with EMA
warmup). Second, we adapt a pretrained **Stable Diffusion v1.5** to the Impressionist style using a
1,500-image WikiArt subset, comparing three fine-tuning strategies — **LoRA** (rank sweep), **full
fine-tuning**, and **DreamBooth** — on image quality (FID), prompt alignment (CLIP score), and
training cost. Throughout, the emphasis is on the empirical *process*: failed runs, their analysis,
and the fixes that followed. Our first evaluation proved unreliable (FID at N=256 could not even
distinguish real paintings from real paintings); after rebuilding it — 2,048 images/model against
2,800 held-out paintings, style-word-free prompts, tuned LoRA scale, and an images-seen-matched full
fine-tune — the corrected result is decisive: **every fine-tune beats base SD, and a 12.8 MB LoRA
(rank 16, scale 1.5, FID 112.8) beats even the fairly-matched 3.44 GB full fine-tune (121.5)** —
style adaptation is a low-rank problem, and parameter-efficient tuning is not merely cheaper but
*better* here.

## 1. Introduction — subject, goal, challenges
**Subject.** Generative modelling of images, specifically learning an artistic *style* (Impressionism)
rather than a single object class.

**Goal.** Two stages: (a) build and train a denoising diffusion probabilistic model (DDPM) from
scratch to understand the noise→denoise mechanism; (b) adapt a large pretrained text-to-image
model (Stable Diffusion v1.5) to generate Impressionist artwork, and compare adaptation strategies.

**Challenges.**
- Style is *global and diffuse* (brushwork, broken colour, soft edges, light) — not localisable.
- Curated style data is small (thousands, not millions) → training high-fidelity from scratch is
  impractical; transfer from a pretrained prior is essential.
- No ground-truth target per prompt → evaluation must be distributional (FID) + perceptual + CLIP.
- Practical: brand-new **RTX 5090 (Blackwell sm_120)** needs bleeding-edge CUDA 12.8 / PyTorch cu128;
  WikiArt has label noise; SD-1.5's original repo was removed from HuggingFace.

## 2. Related Work
Full cited review in [`related_work/related_work.md`](../related_work/related_work.md) (grounded in
real arXiv metadata). In brief: GAN-era image/style synthesis (GANs, StyleGAN, CycleGAN, Neural
Style Transfer) → diffusion fundamentals (Sohl-Dickstein 2015; score-based; DDPM, Ho et al. 2020) →
faster sampling & guidance (DDIM, classifier-free guidance) → **Latent Diffusion / Stable Diffusion**
(Rombach et al. 2022) → personalization & PEFT (DreamBooth, Textual Inversion, **LoRA**). Our work
sits at the intersection of mechanistic from-scratch understanding and a documented PEFT comparison
for a specific artistic style.

## 3. Background — how diffusion works (course framing)
A **forward** process gradually adds Gaussian noise to an image over `T` steps until it is
indistinguishable from noise. A neural network (U-Net) is trained to **reverse** one step at a time,
predicting the noise to remove (ε-prediction). Sampling starts from pure noise and denoises to a new
image. **Latent** diffusion (Stable Diffusion) first compresses the image into a lower-dimensional
latent (via a VAE) that keeps only essential features, then runs the diffusion there — far cheaper,
and the basis for text-to-image generation with a CLIP text encoder + cross-attention.

## 4. Phase 1 — DDPM from scratch
**Implementation** (`src/phase1_ddpm_from_scratch/`, no diffusers for the model): sinusoidal timestep
embedding; U-Net with residual blocks + self-attention at 16×16; cosine β-schedule; ε-prediction MSE;
EMA weights; DDPM + DDIM samplers. ~35.75 M parameters at 64 px.

**Setup.** butterflies-64 (sanity, Phase 1a) then Impressionism-64 (on-theme, Phase 1b).
GPU-resident data; ~510 img/s unthrottled (compute-bound) on the 5090.

**Experiment ladder (show the process).**
- **p1a_butterflies (baseline):** trained from scratch; clear butterflies emerged by ~step 4000, final
  loss ≈ **0.011** (`report/figures/phase1a_loss.png`). ✅
- **Real bug found — EMA dominated by initialization:** at step 4000 the *EMA* samples were pure noise
  while the *raw* model already produced butterflies. Cause: EMA decay 0.9999 averages ~1/(1−decay)=10k
  steps, so at 4k steps the shadow was ≈67% random init. Fix: EMA **warmup** `(1+t)/(10+t)→0.9999` +
  `--reset-ema` (re-seed from good weights). EMA recovered to clean butterflies by step 8000.
  *(A real failure→analysis→fix, superseding the planned synthetic "no-EMA" experiment.)*
- **p1-r01 lr=2e-3 (10× too high):** loss **stalls at ≈0.98** (vs ≈0.01 at lr 2e-4) — gradient clipping
  averts NaN divergence, but the updates overshoot so the model never converges (samples are noise).
  Confirms 2e-4 as the right scale. ✅
- p1-r03 linear-vs-cosine: not run (scope); cosine used throughout, per Improved-DDPM guidance.
- **p1b_impressionism (64px, from scratch):** trained 8k steps (loss ≈0.04). It learns the Impressionist
  **palette and brushwork texture** (broken warm/cool colour) but **not coherent scenes** — 1,200 *diverse*
  paintings (landscapes, portraits, cityscapes) are too varied for a 36 M model at 64 px from scratch, so
  outputs are impressionist-coloured abstract fields. This contrasts sharply with the coherent butterflies
  (a single narrow class) and **motivates Phase 2's pretrained prior**. ✅ *(1st attempt hit a transient
  CUDA crash on step 0; the re-run succeeded.)* Samples: `outputs/phase1/p1b_impressionism/samples/`.

**Results.** Loss curve: `report/figures/phase1a_loss.png`. EMA sample evolution (noise→butterflies,
steps 4k/6k/8k) in `outputs/phase1/p1a_butterflies/samples/`. Sampling used DDIM (50 steps) and ancestral
DDPM; we did not compute FID for the from-scratch models (no Phase-1 eval pipeline) — Phase-1 success is
judged qualitatively and by the loss curve.

## 5. Phase 2 — Fine-tuning Stable Diffusion for Impressionism
**Data** (`src/phase2_sd_finetune/prepare_data.py`): streamed WikiArt, kept `style == Impressionism`
(index 12), center-cropped to 512 px, templated captions (e.g. *"an impressionist painting of a
landscape, in the style of Claude-Monet"*), held-out split for FID. Base model:
`stable-diffusion-v1-5/stable-diffusion-v1-5`.

**Methods compared** (all 512 px, bf16, GPU throttled to ~65 % per the user's request).
- **LoRA** (`train_lora.py`): low-rank adapters on UNet attention; rank sweep **{4, 16, 64}**.
- **Full fine-tuning** (`train_full.py`): all 860 M UNet params, lr 1e-6, gradient checkpointing.
- **DreamBooth** (`train_dreambooth.py`): rare-token instance prompt + generated class images for prior preservation.

**What we ran.** A systematic 6-way comparison (base + LoRA r4/r16/r64 + full-FT + DreamBooth), each
generating 256 images scored on FID + CLIP (§6). Two *planned* failure-ablations — full-FT at lr 1e-4
(catastrophic forgetting) and DreamBooth without prior preservation — were **not run** within the
overnight budget; we prioritised the systematic sweep. (The Phase-1 lr-too-high failure *was* run — §4.)

**Results.** Qualitatively, LoRA r16 produces **strong, coherent Impressionist style transfer** that
follows the prompt (Monet-style landscapes, harbour sunsets, gardens, snowy streets) and refines from
step 250 → 1500 without overfitting (`outputs/phase2/lora_r16/samples/`). The full quantitative comparison
(FID/CLIP across all six models) and the headline same-prompt figure (`report/figures/method_compare.png`)
are in §6.

## 6. Evaluation & comparison

**Methodology.** Each model generates **256 images** from prompts drawn from the training captions
(fixed seed). We report **FID** against the **300 held-out** Impressionism paintings (Inception
features; approximate at this sample size, used as a *relative* score across methods) and a **CLIP
score** (cosine of CLIP image/text embeddings ×100) for prompt alignment, plus trainable-parameter
count and wall-clock per method.

**Results** (auto-filled from `experiments/sweep.log` via `src/common/parse_results.py`).
**⛔ These numbers are retracted — see the banner below the table and the corrected results in §6.1:**

| Model | FID ↓ | CLIP ↑ | Trainable params | Note *(as written at the time — see §6.1)* |
|---|---|---|---|---|
| Base SD-1.5 | 167.43 | 33.48 | 0 | baseline, no fine-tune — ⛔ *v2: base is the **worst** model* |
| LoRA r4 | 159.20 | 33.37 | ~0.8M | "nominal-lowest FID (likely noise)" — ⛔ *v2: 3rd, and the noise call was right* |
| LoRA r16 | 164.07 | 33.15 | ~3.2M | "FID mid-band" — ⛔ *v2: **best model overall*** |
| LoRA r64 | 166.00 | 32.84 | ~12.8M | "no gain over r4, lowest CLIP — rank saturates/overfits" — ⛔ **RETRACTED**: *v2 = 2nd overall, **highest** CLIP* |
| Full FT | 164.10 | 33.49 | 860M | "best CLIP; modest FID gain" — ⛔ *v2: loses to every adapter* |
| DreamBooth | 162.90 | 32.47 | ~3.2M (+prior) | "lowest CLIP (fixed instance prompt)" — ⛔ *measured with `sks` absent from every prompt; see §6.1(9)* |

> **⛔ Everything in §6 up to §6.1 is the *superseded* v1 measurement, retained for the process
> narrative. Do not cite these numbers.** The diagnostic in §6.1 shows this ruler could not tell real
> art from real art (floor 156.7 at this N), and it overturned three readings below: r64 is in fact
> **second-best** overall, r4 is **not** the best, and DreamBooth's "trade-off" was measured with its
> trigger token absent from every prompt. **The corrected results are §6.1.**

**Reading the comparison — as concluded at the time** (figure: `report/figures/method_compare.png`,
also v1-regime: style-word prompts at scale 1.0).
**(1) The methods are close, and FID is a weak discriminator here.** All fine-tunes land in a narrow
**159–167** FID band just below base, with CLIP ~32.5–33.5. This is expected: the prompts already contain
"impressionist painting", so base SD-1.5 is a *strong* baseline and the *added* value of fine-tuning is
partly masked (a cleaner test would drop the style word from prompts — see Limitations). FID at N=256 is
also high-variance, so the ~5-point gaps should not be over-read (notably, rank-4 has the nominal lowest
FID — almost certainly noise, not a real ranking).
**(2) The same-seed visual comparison is more telling than the numbers.** LoRA r4/r16 and full
fine-tuning stay faithful to the base composition while refining brushwork and palette; **LoRA r64
drifts** (it re-composed the scene, adding boats) and **DreamBooth softens/hazes** detail — both adapt
more aggressively, for no metric gain.
**(3) Cost/benefit decisively favours LoRA.** A rank-4–16 adapter (~0.8–3.2 M params, <0.4% of the UNet)
matches **full fine-tuning** (860 M params) on both FID and CLIP. **Takeaway:** for adapting SD to a
*style*, a **small LoRA (rank 4–16) is the best quality-per-parameter choice**; higher rank and full
fine-tuning add cost (and, for r64, drift) without measurable benefit at this data scale.

### 6.1 Evaluation v2 — the corrected measurement (final result)

The v1 numbers above are kept for the process narrative, but a diagnostic showed they are unreliable:
at N=256, **two disjoint halves of the *real* held-out set score FID 156.7 against each other** — a
floor no generator can beat — and every model sat within ~10 points of it. We therefore rebuilt the
evaluation (design + incidents: `JOURNEY.md` §8): **2,048 images per model** vs a **2,800-painting**
held-out reference; **240 neutral prompts with zero style words** (so base SD gets no free hint);
LoRA at the tuned **inference scale 1.5** (with 1.0 as control); and full fine-tuning **retrained
with images-seen matched** to the LoRAs. The floor re-measured at this sample size: **FID 37.6** —
the ruler resolves differences now.

| Model | FID ↓ | above floor | CLIP ↑ |
|---|---|---|---|
| Base SD-1.5 (no fine-tune) | 128.3 | +90.7 | 32.72 |
| LoRA r4 @×1.5 (3.3 MB) | 116.9 | +79.3 | 32.67 |
| LoRA r16 @×1.0 (12.8 MB) | 119.3 | +81.7 | 32.89 |
| **LoRA r16 @×1.5 (12.8 MB)** | **112.8** | **+75.2** | **32.93** |
| LoRA r64 @×1.5 (51 MB) | 114.5 | +76.9 | **33.10** |
| DreamBooth @×1.5, no `sks` in prompts (12.8 MB) | 119.65 | +82.05 | 32.71 |
| DreamBooth @×1.5, **+ `", in sks style"`** (12.8 MB) | 119.66 | +82.07 | **31.66** |
| Full FT, 4 img/step (3.44 GB) | 123.0 | +85.4 | 32.78 |
| Full FT **matched**, 8 img/step (3.44 GB) | 121.5 | +83.9 | 32.84 |

**Reading v2.** (1) **Fine-tuning works**: with no style cue in the prompt, *every* fine-tune beats
base; best gap **−15.5 FID**. (2) **LoRA beats full fine-tuning in the fair fight** — r16@1.5 by
**−8.7 FID** vs the matched full-FT; even the 3.3 MB rank-4 adapter outscores both 3.44 GB models.
The v1 conclusion upgrades from *matches* to **beats at <0.4 % of the parameters**. (3) The
**inference scale alone is worth −6.5 FID** (r16 ×1.0→×1.5, same weights). (4) At matched scale,
**r16 > r4** — v1's "r4 best" was estimator noise, as suspected. (5) Doubling full-FT's images-seen
improved it only +1.5 FID and did not change the verdict: the confound is closed. (6) CLIP is nearly
flat among the style models (32.67–33.10) by design — neutral prompts describe *content* — though two
structured effects survive: rank tracks alignment (r4 32.67 < r16 32.93 < r64 33.10), and the trigger
variant sits a full point lower (point 9). (7)
Honesty: +75 above floor means the best model remains far from indistinguishable from real
Impressionism (unsurprising at 1,200 training images), and v1↔v2 numbers are not comparable
(different prompts and N). (8) **The two initially-omitted models complete the picture** — and
revise a conclusion: LoRA r64 @×1.5 scores **114.5** with the **highest CLIP of all (33.10)** —
second place overall, overturning v1's "r64 is worst" reading (a scale-1.0 + broken-FID artifact;
its composition drift persists qualitatively but barely costs distribution distance). Note this
retracts v1's stronger claim too: r64 **beats** r4 by 2.4 FID, so "16× the parameters for zero gain"
is false; what survives is a **shallow optimum near r16** (116.9 → 112.8 → 114.5).
Net: **every adapter method (3.3–51 MB) beats both full fine-tunes (3.44 GB)**.

**(9) A flaw v1 never tested: DreamBooth's trigger token is inert.** v1 scored DreamBooth on prompts
drawn from the training captions, which never contain `sks` — the method's entire mechanism was
switched off during its own evaluation, and this went unnoticed. We therefore generated the model
**both ways** on identical neutral prompts. The result is unambiguous:

| DreamBooth @×1.5 | FID ↓ | CLIP ↑ |
|---|---|---|
| neutral prompt | 119.65 | 32.71 |
| + `", in sks style"` | 119.66 | **31.66** |

**FID differs by 0.01 — the trigger contributes no style at all — while CLIP falls 1.05.** The
mechanism is clear in hindsight: we bound `sks` using **1,200 instance images under a single fixed
prompt**, so the token never received a contrastive signal distinguishing it from the constant
context. The adapter learned an **unconditional style shift**, and the token merely perturbs the text
embedding away from the content words, costing alignment for nothing. Our configuration was therefore
**style-LoRA training in DreamBooth's clothing** — genuine few-shot DreamBooth binds a handful of
instance images against a class prior, a materially different regime. This reframes point (8): the
style did not "generalize beyond the token"; **it never bound to the token in the first place.**

## 7. Limitations & data-quality notes
- **WikiArt label noise** (observed): some images labelled Impressionism are stylistically off
  (e.g. a Salvador-Dali piece). We filter on the dataset label and accept residual noise.
- From-scratch DDPM at 64 px cannot match SD's fidelity — by design, this motivates Phase 2.
- **Subtle fine-tuning gains:** the prompts contain "impressionist painting", so base SD is already a
  strong baseline and FID/CLIP barely separate the methods — a cleaner experiment would drop the style
  word from the eval prompts; FID at N=256 is also noisy.
  **[Resolved in §6.1: with 240 style-word-free prompts and N=2,048, base becomes the *worst* model
  and the best fine-tune beats it by −15.5 FID. The "subtle gains" were an artifact of prompts that
  handed base SD the answer.]**
- **DreamBooth was evaluated without its trigger token** — v1 drew eval prompts from the training
  captions, which never contain `sks`, so the method's core mechanism was inactive during its own
  measurement. **[Resolved in §6.1(9): tested both ways, the trigger proves inert — ΔFID 0.01,
  ΔCLIP −1.05.]**
- **Remaining, unresolved:** (a) our best model still sits **+75 above the real-vs-real floor**, so it
  is clearly distinguishable from genuine Impressionism — expected at 1,200 training images, but not
  hidden; (b) v1 and v2 numbers are **not comparable** (different prompts *and* N), so only within-v2
  rankings are valid; (c) `base` and `lora_r4` were generated at batch 16 and the rest at batch 8
  (the VRAM fix of D10) — all other settings identical and FID is distributional, so we expect no
  bias, but this is **assumed rather than verified**; (d) `report/figures/method_compare.png` still
  depicts the v1 regime (style-word prompts, scale 1.0) and should be regenerated before publication.
- **The comparison is not matched on images-seen (important):** LoRA trained at batch 2 × grad-accum 4
  (**8 images/step**), whereas full fine-tuning ran at batch 1 × accum 4 (**4 images/step**). At 1,500
  steps each, **full-FT saw only half the image-presentations LoRA did** — so full-FT's parity with LoRA
  may be *understated*. A fair re-run should equalise **images seen**, not optimizer steps.
  **[Resolved in §6.1: the matched rerun improved full-FT only slightly (123.0 → 121.5) and it still
  trails every LoRA — the verdict stands.]**
- **Engineering pitfalls (each diagnosed & fixed):** EMA decay too high (noise samples), a
  torchmetrics/transformers-5 CLIP-score incompatibility, a Windows file-overwrite lock during inference,
  and a transient CUDA crash — see §4 and the experiment log.

## 8. Ethics & responsible use
SD weights are CreativeML OpenRAIL-M; WikiArt used for research/education. Diffusion models can
**memorize** training images (Carlini et al., 2023) — we use a held-out set, watch for overfitting,
and avoid claiming reproduction of specific protected works.

## 9. Conclusion
We built and trained a DDPM **from scratch**, confirming an end-to-end understanding of the
noise→denoise mechanism: it learns recognizable 64×64 images, and the EMA-initialisation pitfall we
hit, diagnosed, and fixed is a concrete instance of the empirical process this project emphasises. We
then adapted Stable Diffusion v1.5 to the Impressionist style and compared LoRA (rank sweep), full
fine-tuning, and DreamBooth. Our first evaluation could not separate the methods — and the reason
became a finding in itself: **FID at N=256 measures its own bias** (real-vs-real floor 156.7, with
every model within ~10 points of it). After rebuilding the measurement (§6.1: 2,048 images/model,
2,800 references, neutral prompts, tuned scale, images-seen-matched full-FT), the corrected verdict
is unambiguous: **every fine-tune beats base SD with no style hint in the prompt, and a 12.8 MB
LoRA (r16 @×1.5, FID 112.8) beats even the fairly-matched 3.44 GB full fine-tune (121.5)** — with
the inference-time LoRA scale alone worth −6.5 FID. Style adaptation is a low-rank problem;
parameter-efficient tuning here is not merely cheaper but **better** — indeed *every* adapter we
trained (3.3–51 MB) outscores *both* 3.44 GB full fine-tunes.

**Repairing the ruler did more than sharpen the numbers — it overturned two conclusions we had
already written down.** LoRA rank 64, dismissed in v1 as the model that "drifts for no gain," is in
fact **second-best overall and the best prompt-follower** (FID 114.5, CLIP 33.10). And DreamBooth's
rare-token binding, whose "trade-off" v1 confidently described, turns out to be **inert**: adding
`sks` changes FID by 0.01 while costing 1.05 CLIP, because 1,200 instance images under a single fixed
prompt never gave the token a contrastive signal to bind to. Both errors were downstream of an
estimator we had never validated — and the second was an experiment we had never actually run.

The methodological lesson therefore stands *above* the result rather than beside it: **validate your
measurement before you trust anything it tells you.** Never quote an FID without its real-vs-real
floor at the same N, and never evaluate a method with its own mechanism switched off. **Future work:**
larger eval sets (5–10k) for stabler FID, a same-seed visual grid regenerated under the neutral-prompt
regime, per-rank schedule tuning, a genuine few-shot DreamBooth comparison (a handful of instance
images against a class prior, as the method intends), ControlNet/IP-Adapter for compositional control,
and a from-scratch *latent* diffusion model to narrow the gap with the pretrained pipeline.

## References
See `related_work/related_work.md`.
