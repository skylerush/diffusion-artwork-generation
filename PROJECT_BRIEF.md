# PROJECT BRIEF — Diffusion Models for Impressionist Artwork Generation
### (This file is the master "goal prompt" / single source of truth for the project. Read it first. Every run, deliverable, and decision traces back to here.)

---

## 0. One-line goal
Train a **diffusion model from scratch** to internalize the noise→denoise mechanism, then **fine-tune a pretrained Stable Diffusion** model to generate **Impressionist-style** artwork — and **document the full empirical process** (failed runs, analysis, hyper-parameter search, improvements) as required by the course rubric.

This is a course final project for a Neural Networks / Deep Learning class at Shenkar. The grade rewards *showing the process*, not a one-shot result.

---

## 1. Operating principles (anti-hallucination / anti-halt contract)
These rules exist because the request was explicitly: *"do this in full precisely and correctly without hallucinations and halts."*

1. **Verify before asserting.** Never state a library version, dataset schema, or API signature from memory. Run the command / inspect the object / fetch the page, then report what was observed. Code comments and the report must reflect *measured* numbers, not guesses.
2. **No fabricated results or citations.** Every metric in the report comes from a logged run. Every cited paper in `related_work/` is grounded in fetched arXiv metadata (real title + arXiv id).
3. **Small-first, then scale.** Prove each pipeline on a tiny subset / few steps before a long run. A 200-step smoke test precedes every multi-hour training.
4. **Checkpoint & resume everything.** Save model + optimizer + step every N steps. Any long run must be resumable so a crash is never a restart-from-zero (avoids "halts").
5. **Determinism where it matters.** Fix seeds; log them. Same config + same seed ⇒ reproducible.
6. **One change at a time.** When tuning, vary a single hyper-parameter per run so cause↔effect is attributable. Log the hypothesis *before* the run and the verdict *after*.
7. **Fail loudly, then analyze.** A diverged/blurry/mode-collapsed run is a deliverable, not a secret. Capture it, explain *why*, state the fix, re-run.
8. **Empirical environment.** The RTX 5090 is Blackwell (sm_120) and needs very recent wheels; pin nothing blindly — install from the cu128 index and confirm with `environment/verify_gpu.py`.

---

## 2. Confirmed environment (measured 2026-06-28)
| Item | Value | Implication |
|---|---|---|
| GPU | **NVIDIA RTX 5090, 32 GB**, compute cap **12.0 (sm_120, Blackwell)** | Needs CUDA **12.8+** + PyTorch **cu128** build. 32 GB allows full SD-1.5 fine-tune at 512px. |
| Driver | 596.21 | Supports CUDA 12.8 runtime via PyTorch wheels. |
| RAM / Disk | 96 GB / ~200 GB free | Ample for datasets + checkpoints. |
| OS / Shell | Windows 11, PowerShell 5.1 + Git Bash | Scripts are PowerShell-first. |
| Python | system has only 3.14 (too new for ML wheels) | We provision **Python 3.12** via `uv` into `.venv`. |
| Internet | pypi / pytorch-cu128 / HuggingFace / arXiv all reachable | Downloads + grounded research OK. |
| Git | 2.43 | Optional local versioning of experiments. |

**Frameworks:** PyTorch (cu128) + HuggingFace `diffusers` / `transformers` / `accelerate` / `datasets` / `peft` (LoRA). FID via `clean-fid` / `torch-fidelity`. Logging via TensorBoard (W&B optional).

---

## 3. Scope — two phases (+ a sanity sub-phase)

### Phase 1 — DDPM **from scratch** (understand the mechanism)
Implement DDPM (Ho et al., 2020) ourselves in PyTorch — forward noising, a U-Net noise predictor (ε-prediction), the β schedule, the reverse sampler — **no diffusers for the model itself**. Goal: prove understanding and produce a working generator on small images.

- **Phase 1a (sanity):** train on an *easy* 64×64 set (e.g. Oxford Flowers / Smithsonian Butterflies) to validate the implementation end-to-end. Success = recognizable samples + decreasing loss.
- **Phase 1b (on-theme):** retrain the same model on a **64×64 Impressionism** subset. Expectation: coherent color/texture "feel" of Impressionism but limited fine detail at this scale — this *motivates* moving to pretrained SD in Phase 2 (a great report narrative about the limits of from-scratch at small compute/scale).

### Phase 2 — **Fine-tune Stable Diffusion** for the Impressionist style
Adapt a pretrained **Stable Diffusion v1.5** to the Impressionist style at **512×512**, comparing three adaptation strategies (this comparison *is* the innovation/contribution):
- **A. LoRA** (rank sweep) on the U-Net (primary; cheap, fast).
- **B. Full fine-tuning** of the U-Net (we have the VRAM; baseline for "best possible" vs cost).
- **C. DreamBooth** and/or **Textual Inversion** (subject/style token; data-efficient).

Deliver a head-to-head: quality (FID), text-alignment (CLIP score), training cost, and qualitative style adherence.

---

## 4. Data plan
**Target style:** Impressionism (WikiArt).

- **Primary source (to verify in the data step, do not assume):** the `huggan/wikiart` dataset on the HuggingFace Hub (fields like `image`, `artist`, `genre`, `style`). Filter `style == "Impressionism"`.
  - **Verify empirically:** load it, print `features`, count rows where style is Impressionism, inspect a few images. If the schema differs or it's gated, fall back to: a Kaggle "Impressionism"/WikiArt mirror, or `keremberke`/other WikiArt mirrors, or a curated artist set (Monet/Renoir/Pissarro/Sisley).
- **Sizes:** Phase 1b ≈ a few thousand images @ 64px. Phase 2 ≈ 1–5k images @ 512px for LoRA/full-FT; 20–200 curated images for DreamBooth/TI.
- **Preprocessing:** center-crop + resize; for Phase 2 build captions. Two captioning options to try: (i) templated from metadata — `"an impressionist painting of {genre}, in the style of {artist}"`; (ii) BLIP-generated captions + a fixed style suffix. DreamBooth uses an instance prompt (e.g. `"a painting in <imp> impressionist style"`) + class/prior images.
- **Held-out eval set:** reserve ~500–1000 Impressionism images, never trained on, as the FID reference.
- **Ethics/licensing (note in report):** WikiArt images are used for research/education; SD weights are under CreativeML OpenRAIL-M. Cite the memorization/copyright literature (training-data extraction) and avoid claiming the model "creates" specific artists' protected works.

---

## 5. Model & hyper-parameters (v1 starting points — to be swept)

### Phase 1 DDPM
- U-Net: base channels 128, channel mults `[1,2,2,2]`, 2 ResBlocks/level, self-attention at 16×16, sinusoidal timestep embedding (+MLP), GroupNorm+SiLU.
- Diffusion: `T=1000`, **cosine** β schedule (also try linear), **ε-prediction**, MSE loss.
- Train: AdamW `lr=2e-4`, batch 128 @64px, EMA decay 0.9999, grad-clip 1.0, mixed precision (bf16), ~100–300k steps (or until samples plateau).
- Sample: ancestral DDPM (1000 steps) and **DDIM** (50 steps) for speed; compare.

### Phase 2 SD-1.5
- Base: **`stable-diffusion-v1-5/stable-diffusion-v1-5`** (verified `200` on HF, 2026-06-28). The original `runwayml/stable-diffusion-v1-5` was removed (`307` redirect). Fallbacks: `botp/stable-diffusion-v1-5`, `CompVis/stable-diffusion-v1-4`. 512×512, bf16.
- **LoRA:** rank sweep `{4, 8, 16, 32, 64}`, α=rank, target attention (`to_q/k/v/out`) (±include feed-forward), `lr≈1e-4`, 1–4k steps, batch 1–4 + grad-accum.
- **Full FT:** `lr≈1e-6…5e-6` (low, to avoid catastrophic forgetting), grad-checkpointing, 1–3k steps; optional 8-bit Adam.
- **DreamBooth:** instance + class images, prior-preservation loss, `lr≈1e-6`, ~800–1500 steps; watch for overfitting/language drift.
- Inference: guidance scale sweep `{3,5,7,9}`, DDIM/Euler, fixed seeds for comparable grids.

---

## 6. "Show the process" — the experiment ladder
Each run = a config (`experiments/<phase>/rNN_<slug>/config.yaml`) + fixed seed + logged loss + sample grid + `notes.md` (**Hypothesis / Result / Decision**). Maintain a master `experiments/RESULTS.md` table.

**Pre-planned instructive failures** (we will actually run these to satisfy "show one try that didn't work, analyze, defend the fix"):
- P1-r01: `lr=2e-3` (10× too high) → loss diverges / pure-noise samples → analyze (gradient blow-up) → drop to 2e-4. 
- P1-r02: **no EMA** → sample quality flickers/worse → add EMA, show side-by-side.
- P1-r03: **linear vs cosine** schedule at 64px → cosine retains more signal late → keep cosine.
- P2-r01: LoRA **rank 4** → underfits, weak style transfer → increase rank, plot style-strength vs rank.
- P2-r02: Full-FT `lr=1e-4` (too high) → catastrophic forgetting (prompts collapse to texture) → drop to ~1e-6 + fewer steps.
- P2-r03: DreamBooth **no prior preservation** → overfits/leaks → add prior images.

Each gets a short written analysis + the corrected re-run. This ladder is the spine of the report.

---

## 7. Evaluation
- **FID** (clean-fid) of generated vs held-out Impressionism set — per run, plotted.
- **CLIP score** (image–text alignment) for Phase 2 prompts.
- **Training/val loss** curves; **sample-evolution** grids over training steps.
- **Qualitative**: fixed-prompt, fixed-seed comparison grids across methods/ranks; a small style-adherence rubric.
- **Cost**: params trained, VRAM, wall-clock per method (LoRA vs full-FT vs DreamBooth).
- (Optional/advantage) **Attention/feature visualization** to "explain" what the model attends to — ties to the course's attention material.

---

## 8. Repository map
```
diffusion-artwork-generation/
  PROJECT_BRIEF.md          <- this file (master spec)
  README.md                 <- quickstart + orientation
  environment/              <- setup.ps1, requirements*.txt, verify_gpu.py, setup.log
  data/                     <- datasets (gitignored); prep scripts live in src
  src/
    common/                 <- shared: schedules, ema, datasets, metrics, viz, seeding
    phase1_ddpm_from_scratch/  <- unet.py, diffusion.py, train.py, sample.py
    phase2_sd_finetune/     <- prepare_data.py, train_lora.py, train_full.py, train_dreambooth.py, infer.py, eval_fid.py
  notebooks/                <- exploratory + figure-generating notebooks
  experiments/              <- per-run configs, logs, notes, RESULTS.md
  outputs/                  <- samples + checkpoints (gitignored)
  related_work/             <- related_work.md (grounded literature review)
  report/                   <- report.md/pdf (the write-up)
  slides/                   <- presentation (subject, goal, challenges)
```

---

## 9. Deliverables (rubric mapping)
| Rubric item | Where |
|---|---|
| Short presentation (subject, goal, challenges) | `slides/` |
| Related work + our innovation | `related_work/related_work.md` |
| Implementation showing the *process* (failures→analysis→improvement) | `src/`, `notebooks/`, `experiments/` + `report/` |
| ≥2 tries, learning curve, hyper-param search | `experiments/RESULTS.md` + report §experiments |
| Our own ideas + comparison to others | Phase-2 method comparison + Phase-1 from-scratch baseline |

---

## 10. Milestones (execution checklist)
- [ ] **M0 Environment**: `uv` → Python 3.12 venv → torch cu128 → deps → `verify_gpu.py` prints `VERIFY_OK sm_120`.
- [ ] **M1 Related work**: grounded `related_work.md` generated and reviewed.
- [ ] **M2 Phase-1 scaffold**: U-Net + diffusion + train/sample; 200-step smoke test passes.
- [ ] **M3 Phase-1a**: train on sanity set; recognizable samples; loss curve logged.
- [ ] **M4 Phase-1 failure ladder**: run P1-r01..r03; analyses written.
- [ ] **M5 Phase-1b**: Impressionism-64 model + samples; document limits.
- [ ] **M6 Data (Phase 2)**: WikiArt Impressionism downloaded, verified, captioned, split.
- [ ] **M7 Phase-2 LoRA**: rank sweep + smoke test → full runs; FID/CLIP logged.
- [ ] **M8 Phase-2 full-FT + DreamBooth/TI**: runs + failure ladder (P2-r01..r03).
- [ ] **M9 Evaluation**: FID/CLIP/cost table + comparison grids.
- [ ] **M10 Write-up**: report + slides assembled from logged artifacts.

---

## 11. Risks & mitigations
- **Blackwell wheels**: `xformers`/`bitsandbytes` may lack sm_120 builds → keep them *optional*; use PyTorch SDPA attention + standard AdamW as the always-works path.
- **SD weight availability/gating**: verify the repo id at run time; have a mirror fallback; accept OpenRAIL-M license.
- **OOM**: bf16 + grad-checkpointing + grad-accum + smaller batch; LoRA before full-FT.
- **Dataset schema drift**: never assume columns — print `features` first; have Kaggle/manual fallbacks.
- **Long-run crashes**: checkpoint+resume; smoke-test first.
- **Overfitting/memorization**: prior preservation, early stop on FID, hold-out set, copyright note.

---

## 12. Quickstart
```powershell
# 1. One-time environment setup (installs uv, Python 3.12, torch cu128, deps; verifies GPU)
powershell -ExecutionPolicy Bypass -File ".\environment\setup.ps1"

# 2. Activate the venv for interactive work
.\.venv\Scripts\Activate.ps1
python environment\verify_gpu.py   # expect: VERIFY_OK sm_120
```
