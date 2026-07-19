# 🪪 MODELS.md — Passports & the Hyper-parameter Story

> Every model we trained, as an ID card — followed by the knob-by-knob account of **what we tuned,
> what happened, and what it cost**. Every number here is verified from `config.json` files, logs,
> and measured artifacts (nothing from memory). The project trained **~20 times** (13 substantive
> runs + 7 smoke/probe runs); this document is what those runs taught us.

---

# Part 1 — Model passports

## Phase 1 — our own DDPM (from scratch, unconditional, 64 px)

Shared anatomy: **our hand-written U-Net** (35.75 M params; base 128, mults [1,2,2,2], self-attention
@16×16, sinusoidal t-embedding) + cosine β-schedule, T=1000, ε-prediction MSE, EMA, bf16, batch 128.

---

### 🦋 `p1a_butterflies` — *the proof the math works*
| | |
|---|---|
| **Trained** | 2026-06-28, two sessions (0→5k, then resumed 5k→8.6k with `--reset-ema`) |
| **Data** | 1,000 Smithsonian butterflies @64px (narrow, easy distribution) |
| **Config** | lr **2e-4** · EMA 0.9999 (+warmup after the fix) · cosine · throttled 0.65 |
| **Result** | final loss **0.0107 → 98.9 % of noise variance explained**; clean, varied butterflies |
| **Scar tissue** | *The EMA bug lived here*: at step 4k the EMA sampled pure noise (shadow ≈67 % random init). Fixed with warmup + reset — without retraining. |
| **Passport stamp** | ✅ the from-scratch mechanism **works** |
| **Checkpoint** | `outputs/phase1/p1a_butterflies/ckpt/last.pt` (572 MB: model+EMA+optimizer) |

### 🎨 `p1b_impressionism` — *the honest limit*
| | |
|---|---|
| **Trained** | 2026-06-29, 8k steps (~47 min; first attempt died in a transient CUDA crash at step 0) |
| **Data** | our 1,200 WikiArt Impressionism paintings @64px (broad, hard distribution) |
| **Config** | identical to p1a — the *only* change is the data |
| **Result** | loss **0.033 → 96.7 % noise explained** — yet samples are **palette-and-brushwork abstractions**, not scenes |
| **Lesson** | low loss ≠ good samples. 1,200 *diverse* paintings overwhelm a 36 M model at 64px; distribution breadth is a hyper-parameter of the *problem*. **This is the measured reason Phase 2 uses a pretrained prior.** |
| **Checkpoint** | `outputs/phase1/p1b_impressionism/ckpt/last.pt` |

### 💥 `p1_r01_lr2e3` — *the deliberate failure*
| | |
|---|---|
| **Trained** | 2026-06-29, 2k steps (13 min) — identical to p1a except **lr ×10 (2e-3)** |
| **Result** | loss **stalls at ≈0.98 → 0.5 % noise explained** = learned nothing; samples are noise |
| **The subtlety** | grad-clipping prevented NaN, so the curve looks "stable" — divergence disguised as a plateau. You must know your loss's floor-for-random (≈1.0) to spot it. |
| **Passport stamp** | ❌ kept deliberately as the control that defends lr=2e-4 |
| **Checkpoint** | `outputs/phase1/p1_r01_lr2e3/ckpt/last.pt` (educational) |

*(Utility runs without passports: `run` = 30-step smoke, `p1a_speedcheck` = the 200→500 img/s
throughput probes that also produced our compute-bound negative result.)*

---

## Phase 2 — Stable Diffusion 1.5 fine-tunes (text-conditioned, 512 px)

Shared anatomy: frozen SD-1.5 (860 M UNet + VAE + CLIP text encoder), bf16, AdamW, DPM-Solver
sampling, trained on the same 1,200 captioned Impressionism paintings (except DreamBooth's captions).

---

### ⚪ `base` — *the reference, not ours*
No training. Already paints "impressionist" **when asked in words** (that's why v1 metrics flattered
it, and why eval-v2 uses neutral prompts). FID v1 167.4 / CLIP 33.5.

### 🌱 `lora_r4` — *the featherweight*
| | |
|---|---|
| **Recipe** | LoRA r=4, α=4 on attention (q,k,v,out) · lr 1e-4 · batch 2×4 = **8 img/step** · 1,500 steps (61 min) |
| **Trainable** | **0.80 M params (0.09 % of the UNet)** → checkpoint **3.3 MB** |
| **Results** | v1 FID **159.2** (nominal best — within metric noise) · CLIP 33.4 |
| **Personality** | faithful to base compositions, gentle style — the "subtle filter" |
| **Verdict** | astonishing value-per-byte; proof that style is a *low-rank* direction |

### ⭐ `lora_r16` — *the flagship*
| | |
|---|---|
| **Recipe** | r=16, α=16 · lr 1e-4 · 8 img/step · 1,500 steps (61 min) |
| **Trainable** | **3.19 M (0.37 %)** → **12.8 MB** |
| **Results** | v1 FID 164.1 / CLIP 33.2 (mid-band); **visually the strongest faithful style** in the same-seed grid. **v2 (honest ruler): FID 112.8 @×1.5 — 🏆 best model in the project**, beating base by −15.5 and the matched full-FT by −8.7 |
| **Hidden depth** | trained at effective scale 1.0 (α=r) — the **inference scale dial** later revealed head-room: ×1.5–2.0 converts even reluctant landscapes; the *style was in the weights all along* |
| **Personality** | period figures, harbour light, painterly texture; keeps compositions intact |
| **Verdict** | the model we'd ship. Prompt it: `generate.py "…"` (defaults to this + scale 1.5) |

### 🍂 `lora_r64` — *the over-achiever that overreaches*
| | |
|---|---|
| **Recipe** | r=64, α=64 · otherwise identical (61 min) |
| **Trainable** | 12.8 M (1.5 %) → **51 MB** |
| **Results** | v1@1.0: FID 166.0, lowest LoRA CLIP; same-seed grid showed **composition drift** (added sailboats). **v2 @×1.5: FID 114.5 with the HIGHEST CLIP of all (33.10) — second place overall.** |
| **Lesson** | v1's "worst LoRA" verdict was a scale-1.0 + broken-ruler artifact — *the narrative bowed to data twice*. Be precise about what survives: **r64 beats r4 (114.5 vs 116.9), so v1's "16× the params for zero measured gain" is retracted.** What stands is a **shallow optimum near r16** (r4 116.9 → r16 112.8 → r64 114.5), not saturation-then-collapse. Drift remains a real qualitative trait — but it costs almost nothing distributionally and comes with the best prompt adherence in the project. |

### 🎭 `dreambooth` — *the "token specialist" that never bound to its token*
| | |
|---|---|
| **Recipe** | LoRA r=16 bound to instance prompt **"a painting in sks impressionist style"** + prior preservation (100 self-generated class images, weight 1.0) · lr 1e-4 · batch 1×4 = **4 img/step** · 1,200 steps (**93 min** — longest, it first generates its own prior set) |
| **Trainable** | 3.19 M → 12.8 MB |
| **Results (v1)** | FID 162.9 · "lowest CLIP (32.5)" — ⚠️ **measured on prompts that never contained `sks`**, i.e. with the method's mechanism switched off. Nobody caught this at the time |
| **Results (v2)** | **no trigger: FID 119.7 · CLIP 32.71** — beats *both* full fine-tunes · **with `", in sks style"`: FID 119.66 · CLIP 31.66** |
| **🔬 The trigger is inert** | FID differs by **0.01** between the two — the token adds **no style whatsoever** — while CLIP drops **−1.05**. It is dead weight that only pulls the text embedding off the content words |
| **Why** | we bound `sks` with **1,200 instance images under one fixed prompt**, so the token never got a contrastive signal separating it from the constant context. The adapter learned an **unconditional style shift**. **This was style-LoRA training in a DreamBooth costume** — real few-shot DreamBooth binds a handful of images against a class prior |
| **Personality** | atmospheric, less literal — but it answers to *everything*, not to `sks` |
| **Fine print** | also trained at 4 img/step (like v1 full-FT) — half the LoRAs' images-seen |

### 🏋️ `full_ft` — *the heavyweight (v1)*
| | |
|---|---|
| **Recipe** | **all 859.5 M UNet params** · lr **1e-6** (100× below LoRA — forgetting insurance) · grad-checkpointing · batch 1×4 = **4 img/step** · 1,500 steps (55 min) |
| **Checkpoint** | **3.44 GB** (269× LoRA r16) |
| **Results** | v1: FID 164.1 · best CLIP 33.5 — indistinguishable from a 12.8 MB LoRA. **v2: FID 123.0 — loses to every LoRA** |
| **Confound** | saw only **half the images** the LoRAs did (4 vs 8 img/step) — discovered post-hoc, hence ↓ |

### ⚖️ `full_ft_matched` — *the fair rematch (born today)*
| | |
|---|---|
| **Recipe** | identical to `full_ft` but batch 2×4 = **8 img/step** — images-seen now equals the LoRAs |
| **Trained** | 2026-07-17 (1 h 52 m; the run survived a VRAM wedge and a midnight reboot around it) |
| **Results (v2)** | FID **121.5** (+83.9 floor) · CLIP 32.84 — better than confounded full_ft (123.0) by only +1.5, and **still behind every LoRA** |
| **Verdict** | the fair fight settled it: the LoRA win is real, not a data artifact. Full fine-tuning is **beaten at 269× the size** |

---

## 🏁 Final scoreboard — evaluation v2 (honest ruler: 2,048 imgs/model, 2,800 refs, zero style words)

Real-vs-real floor at this N: **FID 37.6** (v1's 156.7 floor was estimator bias — see JOURNEY §8).

| Model | FID ↓ | above floor | CLIP ↑ | Size |
|---|---|---|---|---|
| **lora_r16 @×1.5** 🏆 | **112.8** | **+75.2** | **32.93** | 12.8 MB |
| lora_r64 @×1.5 | 114.5 | +76.9 | **33.10** | 51 MB |
| lora_r4 @×1.5 | 116.9 | +79.3 | 32.67 | 3.3 MB |
| lora_r16 @×1.0 | 119.3 | +81.7 | 32.89 | 12.8 MB |
| dreambooth @×1.5 (no `sks`) | 119.7 | +82.1 | 32.71 | 12.8 MB |
| dreambooth @×1.5 (**+ `sks`**) | 119.7 | +82.1 | **31.66** | 12.8 MB |
| full_ft_matched | 121.5 | +83.9 | 32.84 | 3.44 GB |
| full_ft | 123.0 | +85.4 | 32.78 | 3.44 GB |
| base | 128.3 | +90.7 | 32.72 | — |

---

# Part 2 — The hyper-parameter story: what we tuned, what happened, what it cost

> The premise of this project: **you cannot pick these numbers from theory alone.** Each row below
> exists because we trained (or measured) more than once and compared.

### Knobs we swept with multiple training runs

| Knob | Values tried | What happened | **The price** |
|---|---|---|---|
| **Learning rate (scratch DDPM)** | 2e-4 vs **2e-3** | 2e-4 → loss 0.011; 2e-3 → **stalls at 0.98 forever** | Wrong = a whole run learning *nothing* while the curve looks "stable" (clipping hides the blow-up). 13 min bought the proof. |
| **EMA decay** | fixed 0.9999 vs **warmup ramp** | fixed → EMA ≈67 % random init at 4k steps → *noise samples from a good model*; warmup → clean by 8k | Nearly discarded a healthy run as "failed." Rule bought: decay must fit run length (window ≈ 1/(1−d)); warmup makes it safe. |
| **LoRA rank** | **4 / 16 / 64** (3 × 61 min) | metrics within FID noise; r64 drifts & lowest LoRA CLIP; r16 visually strongest-faithful | Rank is linear in params/disk (3 MB→51 MB) and buys *deviation capacity*, not quality. Sweet spot **4–16**. |
| **Adaptation method** | LoRA vs **full-FT** vs **DreamBooth** | all land FID 159–167; full-FT = LoRA quality at **269×** the bytes; DreamBooth trades CLIP for token control | Full-FT: 3.44 GB, forgetting risk, nothing measured in return. DreamBooth: +50 % train time for the prior set. |
| **Images-seen (batch×accum)** | 8 vs 4 img/step (discovered, then **matched rerun**) | v1 full-FT saw *half* the data → confound → `full_ft_matched` | The price of not matching: your comparison table is contestable. One extra 1.5 h run to fix. |
| **LoRA α / inference scale** | trained α=r (scale 1.0); swept **×0→2.5 (+×8)** at inference | 1.0 subtle → **1.5–2.0 converts landscapes** → 2.5 degrades → 8 total breakdown | **Free** — no retraining. The costliest version of this mistake was ours: we *evaluated* the whole v1 at 1.0 and under-sold our own model. |
| **Eval sample size (a hyper-parameter of the *measurement*)** | N=64/128/256 → **2048**; ref 300 → 2800 | same model: FID 227→192→164 by N alone; real-vs-real floor 156.7 at small N | Small-N FID **invalidated our first comparison**. Price of the fix: ~4 h of generation (eval-v2). Cheapest insurance ever vs presenting wrong conclusions. |

### Knobs we set from theory/convention (single value — honest list)
| Knob | Our value | Why we didn't sweep |
|---|---|---|
| β-schedule | cosine | Improved-DDPM's low-res recommendation; linear comparison was planned, cut for budget |
| T (train steps of the chain) | 1000 | DDPM standard; interacts with everything, too expensive to sweep honestly |
| DDIM/DPM inference steps | 50 (scratch) / 25–30 (SD) | quality plateaus; only wall-clock changes |
| Guidance scale | 7.0 | SD community default; sweep planned in brief, not run |
| SD learning rates | LoRA 1e-4 / full 1e-6 | community-established; our budget went to the rank & method sweeps instead |
| U-Net size (scratch) | 35.75 M | fixed so data (p1a vs p1b) was the only variable |

### Throughput/infrastructure knobs (measured, not model-quality)
| Knob | Result |
|---|---|
| RAM-cache decoded images | 200 → 500 img/s ✅ |
| GPU-resident dataset | **no change** → we're compute-bound (negative result, saved further wasted effort) |
| `--max-util` throttle | 98 % → 62 % GPU (×1.5 wall-clock) — the price of sharing the machine |
| bf16 autocast + grad-checkpointing | what makes 860 M trainable in 32 GB |

### The meta-lesson
Three of our biggest findings were **not** on any planned sweep list: the EMA-decay/run-length
interaction, the inference-scale head-room, and the eval-N floor. Multiple training runs don't just
*select* hyper-parameters — they surface the knobs you didn't know existed. That is the argument for
"train multiple times": the price of each extra run is minutes-to-hours; the price of skipping them is
believing a wrong number.

---

*Companion docs: [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) (reference dictionary for every knob —
this file is the **story**, that one is the **definitions**) · [`TUTORIAL.md`](TUTORIAL.md) (drive
the models) · [`JOURNEY.md`](JOURNEY.md) (the full failure→fix narrative) ·
[`experiments/RESULTS.md`](experiments/RESULTS.md) (run log).*
