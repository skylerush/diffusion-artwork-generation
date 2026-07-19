# 🎨 TUTORIAL — playing with the models

> Hands-on guide: which models exist, where they live, how to prompt them, how to train, and how to
> see every metric. **Every command here was executed and verified** on this machine (Git Bash,
> Windows 11, RTX 5090). PowerShell variants noted where the syntax differs.

---

## 0. Open a terminal

```bash
# Git Bash
cd "/c/Users/sagiz/Desktop/Shenkar/neural networks/diffusion-artwork-generation"
```
```powershell
# PowerShell
cd "C:\Users\sagiz\Desktop\Shenkar\neural networks\diffusion-artwork-generation"
```

You **don't need to activate anything** — every command below calls the project's Python directly:

| Shell | Python |
|---|---|
| Git Bash | `./.venv/Scripts/python.exe` |
| PowerShell | `.\.venv\Scripts\python.exe` |

*(The path has a space — keep the quotes on `cd`.)*

---

## 1. ⏱️ 60-second quickstart

```bash
# See every metric in the project (training losses, FID/CLIP tables, model zoo, live v2 status):
./metrics.sh                    # PowerShell: .\metrics.ps1

# Generate an Impressionist image from YOUR prompt (our best model, tuned strength):
./.venv/Scripts/python.exe src/phase2_sd_finetune/generate.py "a harbour with sailing boats at sunset"
# -> SAVED -> outputs/playground/lora_r16_a_harbour_with_sailing_boats_at_sunset_seed0.png
```

The first SD load takes ~20 s (weights are already cached); each 512px image takes ~5–10 s.

---

## 2. 🧠 The model zoo — what exists and where

| `--model` name | What it is | Size | Lives at |
|---|---|---|---|
| `lora_r16` ⭐ default | **Our best LoRA** (rank 16) on SD-1.5 | **12.8 MB** | `outputs/phase2/lora_r16/ckpt/lora_last.pt` |
| `lora_r4` | Tiny LoRA (rank 4) — nearly as good | 3.3 MB | `outputs/phase2/lora_r4/ckpt/lora_last.pt` |
| `lora_r64` | Big LoRA — *drifts* compositions | 51 MB | `outputs/phase2/lora_r64/ckpt/lora_last.pt` |
| `dreambooth` | LoRA trained with the `sks` style token + prior preservation | 12.8 MB | `outputs/phase2/dreambooth/ckpt/lora_last.pt` |
| `full_ft` | **All 860 M UNet params** fine-tuned | 3.44 GB | `outputs/phase2/full_ft/ckpt/unet_last/` † |
| `full_ft_matched` | Full FT, images-seen matched to LoRA (eval-v2 fairness) | 3.44 GB | `outputs/phase2/full_ft_matched/ckpt/unet_last/` † |
| `base` | Plain SD-1.5, no fine-tune (for comparison) | — | HF cache (`~/.cache/huggingface/`, ~4 GB) |

**Phase-1 (from-scratch DDPM, 64px — our own U-Net, no diffusers):**

| Checkpoint | What it makes | Size |
|---|---|---|
| `outputs/phase1/p1a_butterflies/ckpt/last.pt` | clean butterflies (the sanity proof) | 572 MB |
| `outputs/phase1/p1b_impressionism/ckpt/last.pt` | impressionist *palette/texture*, abstract scenes | 572 MB |
| `outputs/phase1/p1_r01_lr2e3/ckpt/last.pt` | noise — the deliberate lr-failure (educational!) | 572 MB |

> A **LoRA is a patch, not a model**: 12.8 MB of low-rank matrices bolted onto the frozen 4 GB base
> at load time. That's the project's headline finding — on the corrected evaluation it *beats* the
> 3.44 GB full fine-tune by 8.7 FID.

**If you cloned this from GitHub**, the four LoRA adapters are already present and work immediately.
The three Phase-1 checkpoints are attached to the [v1.0 release](../../releases/tag/v1.0) — download
one and pass it to `sample.py --ckpt`. The two rows marked † (the full fine-tuned UNets, 3.3 GB
each) are not published; rebuild them with `train_full.py` if you need them.

---

## 3. 🖌️ Prompt it — `generate.py`

```bash
PY=./.venv/Scripts/python.exe    # (PowerShell: $PY = ".\.venv\Scripts\python.exe"; & $PY ...)

# Our best model, tuned strength (defaults: --model lora_r16 --scale 1.5 --steps 30 --size 512):
$PY src/phase2_sd_finetune/generate.py "a woman with a parasol in a garden"

# Compare against plain Stable Diffusion (same seed = same composition):
$PY src/phase2_sd_finetune/generate.py "a woman with a parasol in a garden" --model base

# Stronger style, 4 variations in one grid, different seed:
$PY src/phase2_sd_finetune/generate.py "a snowy village street" --scale 2.0 --n 4 --seed 7

# The full fine-tune (loads 3.4 GB, slower to start):
$PY src/phase2_sd_finetune/generate.py "a field of poppies" --model full_ft

# DreamBooth answers to its trained token 'sks':
$PY src/phase2_sd_finetune/generate.py "a painting in sks impressionist style of a lake" --model dreambooth
```

All flags: `--model` `--scale` `--n` `--steps` `--guidance` `--seed` `--size` `--out`.
Output always lands in `outputs/playground/` (auto-named `model_prompt_seed.png`).

### Verified prompting tips (we tested these)
- **`--scale` is the style dial** (LoRA models only): `1.0` subtle · `1.5` tuned default · `2.0`
  strong · `2.5+` starts degrading · `8.0` = total abstract breakdown (fun to see — proves the
  adapter is live).
- **Conversion is subject- and seed-dependent.** Figures, harbours, gardens convert readily;
  some landscapes (we hit it with a windmill) cling to SD's photo prior even at `--scale 2.0`.
  Try another `--seed` or raise `--scale`.
- **Avoid photography vocabulary** — "golden hour", "4k", "DSLR", "photo" pull toward photographs.
  Painting-friendly words ("at sunset", "broken color") help.
- Saying **"an impressionist painting of …"** makes the style trivially strong. We banned it in our
  *measurements* (it lets base SD cheat), but for playing it's totally fine.

### Ready-made comparison figures
```bash
$PY src/phase2_sd_finetune/demo.py          # base vs our LoRA, 4 prompts, NO style word -> report/figures/demo_base_vs_lora.png
$PY src/phase2_sd_finetune/demo_scale.py    # strength sweep x{0,1,1.5,2,2.5}            -> report/figures/demo_lora_scale.png
```

---

## 4. 🦋 The from-scratch models — `sample.py`

These sample **our own DDPM** (the U-Net we wrote ourselves) — unconditional, 64px, no prompts:

```bash
# 16 butterflies from the Phase-1 model (DDIM, 50 steps):
$PY src/phase1_ddpm_from_scratch/sample.py --ckpt outputs/phase1/p1a_butterflies/ckpt/last.pt

# The Impressionism-64 model (palette & brushwork, abstract scenes — that's the finding):
$PY src/phase1_ddpm_from_scratch/sample.py --ckpt outputs/phase1/p1b_impressionism/ckpt/last.pt --n 25

# See what a diverged model looks like (the deliberate lr=2e-3 failure -> pure noise):
$PY src/phase1_ddpm_from_scratch/sample.py --ckpt outputs/phase1/p1_r01_lr2e3/ckpt/last.pt

# Options: --sampler ddpm (full 1000-step ancestral) | --ddim-steps 50 | --no-ema | --out path.png
```
Default output: next to the checkpoint, `outputs/phase1/<run>/sample_ddim.png`.

---

## 5. 📊 Metrics — `./metrics.sh`

```bash
./metrics.sh          # PowerShell: .\metrics.ps1
```
Works from a terminal **or by double-clicking** — the window pauses at the end instead of closing,
and every run also saves a copy to **`experiments/metrics_latest.txt`** (open it in any editor).

Prints four tables (source: TensorBoard logs + experiment logs + result JSONs):
1. **Phase-1 training runs** — steps, first/final/min loss, and **"% noise explained"** `= (1−MSE)·100`
   (healthy runs ≈ 96–99 %; the deliberate lr-failure ≈ 0.5 %).
2. **Evaluation v1** — FID/CLIP per model, flagged **unreliable** (real-vs-real floor 156.7 at N=256).
3. **Evaluation v2** — the fixed measurement (2048 imgs, neutral prompts, tuned scale); shows **live
   progress** while the sweep runs, full table + *FID-above-floor* once `eval_v2_results.json` exists.
4. **Model zoo** — every checkpoint with its size.

> **Why no accuracy / val-AUC?** Those are *classification* metrics — they need a ground-truth right
> answer. Generative models invent images, so they don't exist here *by construction*. The table
> header explains the honest analogues we use instead.

Extras:
```bash
$PY -m tensorboard.main --logdir outputs/phase1/p1a_butterflies/tb        # live loss curves -> http://localhost:6006
$PY src/common/plot_curves.py --logdir outputs/phase1/p1a_butterflies/tb --tag loss --out loss.png --smooth 3
```

---

## 6. 🏋️ Training — every command

### Phase 1 — the from-scratch DDPM
```bash
# ALWAYS smoke-test first (30 s, tiny model, fake data — validates the whole pipeline):
$PY src/phase1_ddpm_from_scratch/train.py --smoke

# Real run (butterflies-64, ~35.75M params; ~510 img/s unthrottled on the 5090):
$PY src/phase1_ddpm_from_scratch/train.py --data butterflies --run-name my_butterflies \
    --steps 20000 --sample-every 2000 --ckpt-every 1000

# Impressionism-64 (any image folder works as --data):
$PY src/phase1_ddpm_from_scratch/train.py --data data/impressionism_512/train --image-size 64 \
    --run-name my_impressionism --steps 10000

# Reproduce the deliberate failure (lr 10x too high -> loss stalls ~0.98):
$PY src/phase1_ddpm_from_scratch/train.py --data butterflies --run-name fail_lr --lr 0.002 --steps 2000

# Resume after interruption (checkpoints save model+EMA+optimizer+step):
$PY src/phase1_ddpm_from_scratch/train.py --data butterflies --run-name my_butterflies \
    --resume outputs/phase1/my_butterflies/ckpt/last.pt
#   add --reset-ema if the EMA was contaminated (that's how we fixed the famous EMA bug)

# Sharing the GPU? throttle to a duty cycle (0.65 = keep ~35% free for you):
#   --max-util 0.65
```
Each run writes to `outputs/phase1/<run-name>/`: `config.json`, `tb/` (TensorBoard),
`samples/ddim_{ema,raw}_*.png` every `--sample-every`, `ckpt/last.pt`.

### Phase 2 — fine-tuning Stable Diffusion (needs `data/impressionism_512/`, already built)
```bash
# (Data prep, only if rebuilding: streams WikiArt style==Impressionism -> 512px JPGs + captions)
$PY src/phase2_sd_finetune/prepare_data.py --max-images 1200 --holdout 300

# LoRA (~1 h at full speed; every trainer also has --smoke for a 6-step validation):
$PY src/phase2_sd_finetune/train_lora.py --rank 16 --steps 1500 --run-name my_lora_r16
#   knobs: --rank {4,16,64} --alpha (default=rank; alpha=2*rank trains a stronger scale-2 adapter)
#          --lr 1e-4 --batch-size 2 --grad-accum 4 --max-util 0.65 --smoke

# Full fine-tune (all 860M params, low lr to avoid catastrophic forgetting; ~1-1.7 h):
$PY src/phase2_sd_finetune/train_full.py --steps 1500 --lr 1e-6 --batch-size 2 --grad-accum 4 \
    --run-name my_full_ft

# DreamBooth (instance token 'sks' + generated prior images):
$PY src/phase2_sd_finetune/train_dreambooth.py --with-prior --steps 1200 --run-name my_dreambooth
```
Each run writes `outputs/phase2/<run-name>/`: `config.json`, `samples/` (4 fixed prompts every
`--sample-every`), `ckpt/lora_last.pt` (or `ckpt/unet_last/` for full-FT). Training progress prints
`step | loss | img/s` lines — that loss is the ε-MSE (see §5 for how to read it).

---

## 7. 🔬 Evaluation — measuring a model properly

```bash
# Generate an eval batch (2048 for trustworthy FID; --neutral = zero style words in prompts):
$PY src/phase2_sd_finetune/infer.py --lora outputs/phase2/my_lora_r16/ckpt/lora_last.pt --rank 16 \
    --lora-scale 1.5 --neutral --n 2048 --batch 16 --steps 25 --run-name my_lora_r16

# Score it (FID vs real held-out paintings + CLIP prompt-adherence):
$PY src/phase2_sd_finetune/eval_v2.py \
    --ref-dirs data/impressionism_512/heldout data/impressionism_512_ref/heldout \
    --models "mine=outputs/phase2/my_lora_r16/eval_samples"

# Why 2048? See the proof that small-N FID lies (real-vs-real floor demo):
$PY src/phase2_sd_finetune/fid_diagnostic.py
```
**Iron rule from this project:** never trust an FID without its **real-vs-real floor** at the same
sample size. `eval_v2.py` computes it automatically and reports each model as **FID-above-floor**.

*(The one-command version of all of this — retrain-matched + 6 models × 2048 + auto-eval — is
`environment/run_eval_v2.ps1`, which is what's running as the `diffusion_eval_v2` scheduled task.)*

---

## 8. ⚠️ Gotchas

- **Quote the project path** — it contains a space (`neural networks`).
- The HF warning `You are sending unauthenticated requests` is **benign** (public models).
- **Right now** the v2 evaluation sweep owns most of the GPU (Task Scheduler job `diffusion_eval_v2`,
  ~5 h). Playing alongside it works (we did) but is slower. Check progress anytime: `./metrics.sh`
  or `tail experiments/eval_v2.log`.
- Long runs: prefer `--max-util 0.65` when you need the machine; every trainer checkpoints, and
  Phase-1 resumes with `--resume` (+ `--reset-ema` if needed).
- Windows + spaces + bash: when a `--out` path has spaces, quote it.

---

## 9. Where to read more

| File | What |
|---|---|
| `PROJECT_BRIEF.md` | The master spec (goal, phases, rules) |
| `JOURNEY.md` | The full process story — 12 failures → diagnoses → fixes |
| `report/report.md` | The academic write-up |
| `experiments/RESULTS.md` | Run-by-run experiment log |
| `slides/slides.md` | The presentation (`marp slides/slides.md --pdf`) |
