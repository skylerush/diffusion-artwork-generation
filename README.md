# Diffusion Models for Impressionist Artwork Generation

Sagie Zaoui — Shenkar College, Neural Networks final project.

The project has two parts. The first implements a DDPM from scratch in PyTorch — U-Net noise
predictor, cosine β-schedule, ε-prediction, EMA, and both the ancestral DDPM and DDIM samplers —
and trains it at 64×64, to work through the noise→denoise mechanism directly rather than through a
library. The second fine-tunes Stable Diffusion v1.5 on 1,200 WikiArt Impressionism paintings at
512×512 and compares three adaptation strategies: LoRA at ranks 4, 16 and 64, full U-Net
fine-tuning, and DreamBooth.

![Eight models on the same prompts and seeds](report/figures/hero_grid_8models.png)

## Results

Each model generated 2,048 images from 240 prompts containing no style words, scored against 2,800
held-out paintings. Two disjoint halves of the real reference set score FID 37.6 against each other;
that is the floor everything else is measured against.

| Model | FID | Above floor | CLIP | Size |
|---|---|---|---|---|
| LoRA r16, scale 1.5 | 112.8 | +75.2 | 32.93 | 12.8 MB |
| LoRA r64, scale 1.5 | 114.5 | +76.9 | 33.10 | 51 MB |
| LoRA r4, scale 1.5 | 116.9 | +79.3 | 32.67 | 3.3 MB |
| DreamBooth | 119.7 | +82.1 | 32.71 | 12.8 MB |
| Full fine-tune, images-seen matched | 121.5 | +83.9 | 32.84 | 3.44 GB |
| Base SD-1.5, no fine-tune | 128.3 | +90.7 | 32.72 | — |

The 12.8 MB adapter beats the 3.44 GB full fine-tune by 8.7 FID while training 0.37% as many
parameters, and every adapter here beats both full fine-tunes. Raising the LoRA scale from 1.0 to
1.5 at inference is worth another 6.5 FID and costs nothing to apply.

The first version of this evaluation used 256 images per model. At that sample size two sets of
genuine Impressionist paintings score FID 156.7 against each other, so the metric could not separate
real art from real art, never mind rank six models. Rebuilding it reversed two conclusions that had
already been written up — rank 64 went from worst LoRA to second best overall — and turned up an
experiment that had never actually been run: DreamBooth had only ever been scored on prompts that
omit its own trigger token. `JOURNEY.md` §8 covers this in full.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File ".\environment\setup.ps1"
.\.venv\Scripts\Activate.ps1
python environment\verify_gpu.py   # expect: VERIFY_OK sm_120
```

Built and measured on an RTX 5090 (32 GB, Blackwell sm_120), Windows 11, Python 3.12 via `uv`,
PyTorch cu128. `environment/setup.log` has the install transcript.

## Usage

Generate an image with the fine-tuned model:

```bash
python src/phase2_sd_finetune/generate.py "a harbour with sailing boats at sunset"
```

Print every metric in the project, read from the committed logs and result files:

```bash
./metrics.sh        # PowerShell: .\metrics.ps1
```

`TUTORIAL.md` covers prompting, training, and evaluation in more detail.

## Layout

| Path | Contents |
|---|---|
| `PROJECT_BRIEF.md` | Master spec: goal, plan, experiment ladder, deliverables |
| `TUTORIAL.md` | How to prompt the models, train, and read the metrics |
| `MODELS.md` | Per-model passports and the hyper-parameter story |
| `HYPERPARAMETERS.md` | Reference for every knob: what it means and what we set it to |
| `JOURNEY.md` | Roadmap, timeline, and every failure with its diagnosis and fix |
| `src/phase1_ddpm_from_scratch/` | The from-scratch DDPM: U-Net, diffusion, train, sample |
| `src/phase2_sd_finetune/` | SD fine-tuning: LoRA, full, DreamBooth, inference, evaluation |
| `src/common/` | Schedules, EMA, seeding, metrics, plotting |
| `experiments/` | Run logs, result JSONs, `RESULTS.md` |
| `outputs/` | Sample grids, TensorBoard scalars, per-run configs |
| `related_work/` | Literature review |
| `report/`, `slides/` | Write-up and presentation |

The working tree is about 20 GB, most of it model checkpoints, the dataset, and 18,432 evaluation
images. Those are left out here since they can all be regenerated — `prepare_data.py` rebuilds the
dataset and the training scripts rebuild the weights. The LoRA adapters are small enough to include,
so `generate.py` works straight after a clone. Sample grids, figures, logs, TensorBoard scalars and
per-run configs are all committed, so every artifact the write-ups cite resolves.
