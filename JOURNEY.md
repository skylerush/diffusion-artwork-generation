# The Journey — Process, Roadmap, Difficulties & Improvements

> **This is the "show the process" deliverable.** The brief asks not for a one-shot implementation but
> for the *process*: what failed, how we analysed it, what we tried to improve, and why we made the
> decisions we made. Every claim below is backed by an artifact (log, figure, or sample) — nothing here
> is reconstructed from memory. Timestamps come from file mtimes and `experiments/sweep.log`.

---

## 1. Roadmap — what we set out to do, and why

| Stage | Goal | Rationale |
|---|---|---|
| **Phase 1** | Implement & train a DDPM **from scratch** (no diffusers for the model) | Prove we understand the noise→denoise mechanism, not just the API |
| **Phase 1a** | Train on an *easy* set (butterflies-64) | A narrow distribution: if this fails, the implementation is wrong |
| **Phase 1b** | Retrain on **Impressionism-64** | On-theme, and (we predicted) it would *expose the limits* of from-scratch |
| **Phase 2** | Fine-tune **Stable Diffusion v1.5** on WikiArt Impressionism | Where the real artwork comes from |
| **Phase 2 sweep** | Compare **LoRA (rank sweep) vs full fine-tune vs DreamBooth** | The contribution: a controlled, documented method comparison |

**Deliberate design choice:** validate small before scaling. Every pipeline got a *smoke test*
(tiny model, few steps, sometimes fake data) before any multi-hour run. This caught nothing dramatic —
which is the point: it's cheap insurance, and it let us trust the long runs.

---

## 2. Timeline — what actually happened

### Build phase (18:53 → 21:46, ~3 h)
| Time | Event | Outcome |
|---|---|---|
| 18:53 | Environment install + literature review generated **in parallel** | ✅ torch 2.11.0+**cu128** on RTX 5090 (sm_120) |
| 19:05 | Phase-1 **smoke test** (16 px, 30 steps, fake data) | ✅ whole pipeline validated in <1 min |
| 19:11 | First real butterflies run (300 steps) | ✅ data path + HF download work |
| 19:16 | **Throughput investigation** (`p1a_speedcheck`) | Two optimisations → one *negative result* (see D2) |
| 19:43–19:56 | Butterflies training… **EMA samples are pure noise** | 🔴 **The flagship bug** (see D1) |
| 20:06–20:07 | EMA **warmup fix** + `--reset-ema` written | ✅ fix applied *without losing 5,000 trained steps* |
| 20:08 | Butterflies **resumed** from step 5000 with corrected EMA | ✅ |
| 20:15 | First post-fix samples (raw **and** EMA saved side-by-side) | ✅ EMA recovering |
| 20:28 | **EMA fully recovered** → clean butterflies (step 8000) | ✅ Phase 1 proven |
| 20:34 | **LoRA r16** — first real Impressionist fine-tune launched | ✅ gorgeous output by step 250 |
| 21:46 | **Overnight sweep launched** | → |

### Overnight sweep (21:46 → 02:56, **5 h 10 m**)
| Time | Run | Duration | Result |
|---|---|---|---|
| 21:46 | `infer/eval base` | 5 m | FID 167.43 · CLIP 33.48 |
| 21:51 | `infer lora_r16` | — | 🔴 **CRASHED** (`PermissionError`, see D4) — caught live, fixed mid-sweep |
| 21:52 | `train lora_r4` | **61 m** | FID 159.20 · CLIP 33.37 |
| 22:58 | `train lora_r64` | **61 m** | FID 166.00 · CLIP 32.84 |
| 00:05 | `train full_ft` | **55 m** | FID 164.10 · CLIP 33.49 |
| 01:04 | `train dreambooth` | **93 m** (longest — it also generates 100 class images first) | FID 162.90 · CLIP 32.47 |
| 02:42 | `train p1_r01_lr2e3` | 13 m | ✅ the deliberate failure (see D3) |
| 02:55 | `train p1b_impressionism` | 35 s | 🔴 **native crash** (see D7) |
| 02:56 | **SWEEP DONE** | | |

### Assembly (02:56 → 03:06)
Re-inferred r16 (its real metrics), built the comparison figure (03:03), re-ran p1b successfully (03:06),
then **corrected the report** when the data contradicted an earlier claim (see D8).

### Post-hoc
Discovered a **177 GB runaway log** the sweep script had been writing (see D9). Fixed; 177 GB reclaimed.

---

## 3. The difficulties — every problem, diagnosed and resolved

### 🔴 D1 — The EMA produced *pure noise* (the flagship failure)
**Symptom.** At step 4000 the training loss had fallen to ~0.05 (clearly learning), yet the sampled
images were **pure RGB static**. Complete disconnect between loss and samples.

**How we analysed it.** Rather than guess, we isolated the variable: we sampled the **raw (non-EMA)
model** from the same checkpoint. It produced **clean, recognisable butterflies**. So the U-Net was
fine — the *EMA shadow weights* were the problem.

**Root cause (the maths).** EMA averages over a window of ≈ `1/(1−decay)` steps. With `decay = 0.9999`
that's **10,000 steps** — but we were only at step 4,000. The shadow was therefore still
`0.9999⁴⁰⁰⁰ ≈ 0.67` → **≈67 % random initialisation**. We were sampling a model that was two-thirds
noise. A decay tuned for the DDPM paper's *hundreds of thousands* of steps is simply wrong for a short run.

**The fix.** EMA **warmup**: ramp the decay as `min(max_decay, (1+t)/(10+t))`, so it tracks fast early and
smooths late. Plus a `--reset-ema` flag to **re-seed the shadow from the good weights on resume** — which
let us apply the fix **without throwing away the 5,000 steps already trained**.

**Result.** EMA recovered to clean butterflies by step 8000, and now looks *smoother* than the raw model —
exactly what EMA is supposed to do.
**Evidence:** `outputs/phase1/p1a_butterflies/samples/` — `ddim_0004000.png` (noise) vs
`ddim_ema_0008000.png` (butterflies); fix in `src/common/ema.py`.

> **Why this matters:** this failure *superseded* a synthetic "no-EMA" ablation we had planned. A real bug,
> honestly diagnosed, beats a staged one.

---

### 🟡 D2 — Two optimisation attempts, and an instructive **negative result**
**Symptom.** Training ran at only **~200 img/s** — far too slow for an RTX 5090.

**Attempt 1 (hypothesis: data-bound).** `num_workers=0` meant every image was PIL-decoded on the main
thread each step. → **Cached decoded images in RAM** as a uint8 tensor. Result: **~200 → ~500 img/s.** ✅

**Attempt 2 (hypothesis: still data-bound).** Moved the **entire dataset GPU-resident** (uint8 on the GPU;
index + normalise + flip on-device, zero CPU work per step). Result: **no improvement (~510 img/s).** ❌

**What we learned (the valuable part).** The second attempt *failing* was informative: it proved we were
**compute-bound on the U-Net**, not data-bound. We therefore **stopped optimising** rather than burn hours
chasing a bottleneck that wasn't there. (`torch.compile` was the remaining lever, but it's unreliable on
Windows + Blackwell, so we consciously declined it.)

> A negative result is still a result — and knowing *when to stop* optimising is a real engineering decision.

---

### 🟡 D3 — The deliberate failure: learning rate 10× too high
**Setup.** `p1_r01_lr2e3` — identical to the working run, but `lr = 2e-3` instead of `2e-4`.

**Result.** The loss **stalls at ≈0.98** and never descends (the working run reaches **≈0.011**). Notably it
did *not* explode to NaN — gradient clipping (`clip_grad_norm=1.0`) prevented that — but the updates
**overshoot the minimum every step**, so the model never converges. Samples are noise.

**Defence of the decision.** This confirms `2e-4` (the DDPM-paper scale) is right, and demonstrates that
*grad-clipping masks divergence into stagnation* — the loss curve looks "flat and stable," which is a
deceptively benign-looking failure mode. **Evidence:** `outputs/phase1/p1_r01_lr2e3/`, `sweep.log`.

---

### 🔴 D4 — A Windows file-lock killed inference **mid-sweep** (caught live)
**Symptom.** 5 minutes into the 5-hour unattended sweep, `infer:lora_r16` exited 1 with
`PermissionError: [Errno 13]` on `gen_00000.jpg`.

**Why we caught it.** We had armed a **monitor** on the sweep log filtering for `FID/CLIP/Traceback/exit≠0`
— specifically so a silent failure couldn't rot for 5 hours.

**Diagnosis.** The script tried to **overwrite** stale images left by an earlier validation run; the indexed
Desktop had a transient lock on the old file. Crucially: **only r16 was affected** (its directory was
pre-populated); every other model writes to a *fresh* directory, so it creates rather than overwrites.

**Fix, applied live.** Hardened `infer.py` (clear stale files first + retry-on-`PermissionError`). Because
each sweep step spawns a *fresh* Python process, the **remaining models automatically picked up the fixed
script** — the sweep healed itself mid-flight. r16 was re-inferred afterwards.

---

### 🟡 D5 — CLIP score crashed: a library incompatibility
**Symptom.** `torchmetrics.CLIPScore` → `AttributeError: 'BaseModelOutputWithPooling' object has no
attribute 'norm'`.

**Diagnosis.** `torchmetrics`' CLIPScore is incompatible with **transformers 5.x** (the model now returns a
structured output object where a tensor was expected).

**Fix.** Bypassed it — computed CLIP score **directly**: normalise CLIP's image and text embeddings, take the
cosine, ×100. Fewer dependencies, and we now know exactly what the number means. **Evidence:**
`src/phase2_sd_finetune/eval.py`.

---

### 🟡 D6 — Stable Diffusion 1.5 had been **deleted from HuggingFace**
`runwayml/stable-diffusion-v1-5` — the canonical repo every tutorial names — returns **307** (removed in
2024). Instead of assuming, we **probed candidates** and found `stable-diffusion-v1-5/stable-diffusion-v1-5`
returns **200**. Verified before building on it. *(Rule applied: verify, never assume a version or a URL.)*

---

### 🟡 D7 — `p1b` died with a native access violation
`exit -1073741819` (`0xC0000005`) **35 seconds in** — after the model and all 1,200 images had loaded, i.e.
on the *first training step*. No Python traceback → a native/CUDA-level crash. Since the code path was
**byte-identical** to the butterflies run that had worked for hours, we judged it a **transient CUDA fault**
after a night of back-to-back GPU jobs, and simply **re-ran it — which succeeded.** ✅

---

### 🔵 D8 — We **overclaimed**, then retracted it when the data arrived
Before r16's true metrics existed, the draft asserted *"LoRA r16 is the sweet spot — clearly the best."*
When the real number landed (**FID 164.07**), it was **mid-band** — rank 4 actually had the *lowest* FID
(159.20). We **rewrote the analysis** to the honest conclusion (all methods cluster; FID is a weak
discriminator here; the real result is cost/benefit), and fixed a table note that still called r4
"underfit" when it had in fact scored best. We also found the report listed **failure-ablations we never
ran** (full-FT @ lr 1e-4; DreamBooth without prior) and corrected it to say so explicitly.

> The data disagreed with the narrative, so **the narrative changed.**

---

### 🔴 D9 — A **177 GB** runaway log
The sweep script's final line did `Select-String -Path $log ... | Out-File $log -Append` — **reading and
appending the same file**, an infinite self-feed. It grew `sweep.log` to **177 GB**, silently consuming the
disk (free space fell to 63.5 GB). It also explains the earlier "monitor flood" and a spurious exit code.
**Fixed:** salvaged the real 835-line log, deleted the runaway file, and patched the script to write its
summary to a **separate** file. **177 GB reclaimed** (63.5 → 240.7 GB free).

---

### 🔴 D10 — A VRAM oversubscription **wedge**: 35 minutes became 5 h 45 m
**Symptom.** During the v2 sweep, `gen:lora_r16_s1.0` (batch 16) ran from 17:05 to **22:50** — five
hours forty-five minutes for a job that takes ~35 minutes — and then exited **−1**. No traceback.

**Diagnosis.** The batch-16 generation job and a busy interactive desktop together oversubscribed the
32 GB. The driver began **paging VRAM to host memory**; throughput collapsed by roughly **65×** and
the job eventually wedged. The tell was the *shape* of the failure: not a clean OOM exception, but a
process still "running" while producing almost nothing.

**Fix.** Generation batch **16 → 8** (`environment/run_eval_v2_resume.ps1`), halving VRAM pressure.

**Confirmation — the diagnosis is clean.** The identical job, re-run at batch 8:
**14:49:27 → 15:24:18 = 35 minutes, exit 0.** Same model, same prompts, same everything but batch.

> **Lesson.** An OOM that *throws* is a good day. The dangerous failure is the one that silently
> degrades — a job that is technically alive, making 1.5 % of its normal progress. Wall-clock per
> unit of work is a health metric, not just a performance metric.

---

### 🔴 D11 — A machine **reboot** killed the sweep at 00:35
**Symptom.** `gen:full_ft_matched` began at 00:10 and was killed mid-flight at **1056 / 2048 images**
by a machine reboot.

**Fix.** `environment/run_eval_v2_final.ps1` — a post-reboot final leg that regenerates
`full_ft_matched` (our hardened `infer.py` clears the partial folder first, so a half-written
directory is not a trap) and then `lora_r16_s1.0`, before running the evaluation.

> **The fourth and fifth resilience lessons of the project**, after checkpoint/resume for training
> and resume-by-skip for network streams: *any* unattended multi-hour job needs to be re-runnable
> from partial state. The gap-closing leg (`run_eval_v2_gap.ps1`) took this further — it **skips any
> model directory that already holds 2,048 images**, so a relaunch costs nothing for completed work.

---

### Smaller obstacles (each verified, each fixed)
| Problem | Resolution |
|---|---|
| **Python 3.14** (Windows Store stub) — ML wheels lag new Python; pip sandboxed | Provisioned **Python 3.12** via `uv` |
| **RTX 5090 is Blackwell (sm_120)** — ordinary torch wheels won't run at all | Installed from the **cu128** index; verified with a real GPU matmul |
| `uv` install failed once (`Access denied`, os error 5) | Transient file lock (checked: *not* OneDrive, Defender off) → retried → clean |
| WikiArt prep looked "stalled" (0 images for minutes) | Diagnosed: one-time **parquet shard download**, not an ordering problem (20 hits in 86 rows = **23%**) |
| Caption bug: *"a painting of a **Unknown Genre**"* | Fixed the genre filter |
| **WikiArt label noise** — a *Salvador Dalí* piece labelled Impressionism | Documented as a data-quality caveat; we filter on the label and accept residual noise |
| Background Python **buffered its stdout** (logs looked empty) | Switched to `python -u` |

---

## 4. Improvements — what we tried, and what it bought us

| # | Attempt | Hypothesis | Result | Decision |
|---|---|---|---|---|
| 1 | Cache decoded images in RAM | Data-bound | **200 → 500 img/s** ✅ | Keep |
| 2 | Whole dataset **GPU-resident** | Still data-bound | **No change** ❌ | Revert the assumption: we're **compute-bound**. Stop optimising. |
| 3 | Fused `_foreach` EMA update | EMA kernel overhead | Negligible | Keep (harmless, cleaner) |
| 4 | **EMA warmup** | Fixed decay too high for short runs | **Noise → clean samples** ✅ | The single highest-impact fix |
| 5 | `--reset-ema` on resume | Salvage the corrupted shadow | Fix applied **without losing 5,000 steps** ✅ | Keep |
| 6 | **GPU duty-cycle throttle** (`--max-util`) | User needed the GPU | **98 % → 62 %** (verified via `nvidia-smi dmon`) ✅ | Kept for every run |
| 7 | Save **raw *and* EMA** samples | Never be blind to which model is broken | Made D1's recovery visible | Keep |
| 8 | Harden `infer.py` (clear + retry) | Windows lock | Sweep healed mid-flight ✅ | Keep |
| 9 | Manual CLIP score | Library incompatible | Works, fewer deps ✅ | Keep |

---

## 5. Hyper-parameter search — what we scanned and what we found

| Hyper-parameter | Values tried | Finding |
|---|---|---|
| **Learning rate (Phase 1)** | `2e-4` vs **`2e-3`** | 10× too high → loss **stalls at 0.98**, never converges. `2e-4` confirmed. |
| **EMA decay** | `0.9999` fixed vs **warmup ramp** | Fixed decay = noise on short runs. **Warmup is essential** below ~50k steps. |
| **Noise schedule** | **cosine** (chosen) | Cosine per Improved-DDPM; retains signal longer at low resolution. |
| **LoRA rank** | **4 / 16 / 64** | **Saturates.** r4 ≈ r16 on metrics; **r64 is worse** (drifts, lowest CLIP). More capacity ≠ better. |
| **SD learning rate** | LoRA `1e-4`; full-FT **`1e-6`** | Full-FT needs a ~100× lower LR to avoid catastrophic forgetting. |
| **Adaptation method** | LoRA / full-FT / DreamBooth | **Small LoRA matches full-FT at <0.4 % of the parameters.** |

---

## 6. Key decisions, and how we defend them

1. **Smoke-test everything before long runs.** Cheap; it's why 5 hours of unattended sweep didn't rot.
2. **Sample raw *and* EMA.** Born from D1 — never be unable to tell *which* model is broken.
3. **Resume-with-reset instead of restart-from-zero.** The EMA fix preserved 5,000 GPU-steps of work.
4. **Stop optimising after the negative result (D2).** Knowing when a bottleneck *isn't* there is as
   valuable as finding one.
5. **Monitor the unattended sweep for failure signatures, not just success.** It's what caught D4 within
   minutes instead of after 5 wasted hours.
6. **Throttle the GPU to 65 %** — the user needed the machine. Correctness of the science was unaffected
   (identical steps/data); only wall-clock changed.
7. **Let the data overrule the narrative (D8).** The most important decision in the whole project.

---

## 7. What we'd do differently — open issues we are *not* hiding

1. **The prompts leak the answer.** Our eval prompts literally contain *"an impressionist painting…"*, so
   **base SD-1.5 is already a strong baseline** — which masks how much the fine-tuning actually adds and is
   the main reason all methods cluster at FID 159–167. **The single best next experiment:** re-run the eval
   with the style word **removed** from the prompts.
2. **The comparison isn't perfectly matched on images-seen.** LoRA ran at batch 2 × grad-accum 4 (**8
   images/step**) while full-FT ran at batch 1 × accum 4 (**4 images/step**). At 1,500 steps each, **full-FT
   saw half as many image-presentations as LoRA.** Full-FT's parity with LoRA may therefore be *understated*.
   A fair rerun should equalise images-seen, not steps.
3. **FID at N=256 is noisy** — the ~5-point gaps should not be over-read. Needs ≥2k images to be trustworthy.
4. **Two planned ablations were not run** (full-FT @ lr 1e-4 → catastrophic forgetting; DreamBooth without
   prior preservation) — cut for the overnight budget; the systematic sweep was prioritised.
5. **Phase-1 has no FID.** From-scratch quality was judged qualitatively + by loss curve.

---

## 8. Round 2 — fixing the evaluation (the most important experiment in the project)

After the project was "finished", we asked a simple question: **are our metrics actually trustworthy?**
The answer was no — and chasing it produced the strongest result in the whole project.

### 8.1 The discovery: our ruler was broken
FID fits a Gaussian to **2048-dimensional** InceptionV3 features and compares means + covariances.
Estimating a 2048×2048 covariance needs N ≫ 2048. **We used N = 256** → rank ≤ 255. So we ran the
decisive diagnostic (`src/phase2_sd_finetune/fid_diagnostic.py`):

| Test | Result |
|---|---|
| **REAL vs REAL** — 150 genuine Impressionist paintings vs 150 *other* genuine paintings | **FID = 156.7** |
| our six models (v1) | FID = **159 – 167** |
| same model, N = 64 → 128 → 256 | FID = **227 → 192 → 164** |

**Two real Monets score 156.7 against two other real Monets.** Every model we trained landed within
~10 points of a floor that a *perfect* generator could not beat — and the *same model's* score improved
by **63 points** just from adding samples.

> **Our v1 FID was measuring the estimator's own bias, not our models. The famous "all methods cluster
> tightly" finding was an artifact of a broken ruler.** This retroactively invalidates the *quantitative*
> half of the v1 comparison (the qualitative/visual half stands).

### 8.2 Second discovery: we were under-applying our own model
A LoRA contributes `scale · B@A`, and `scale` is a **free knob at inference** (peft's `layer.scaling`;
default = `lora_alpha / r`). We trained with **α = r → scale 1.0**, but the common convention is
**α = 2r → scale 2.0**. A scale sweep (`demo_scale.py`, figure `demo_lora_scale.png`) showed that at
**×1.5–2.0 the landscapes finally convert** — visible brushwork, photographic detail dissolving into
atmosphere — while ×2.5 begins to degrade.

> **We shipped the whole evaluation at scale 1.0. The style was in the weights the entire time; we just
> never turned it up.** Costs nothing to fix.

### 8.3 The v2 design

| | v1 (what we did) | **v2 (the fix)** | Why |
|---|---|---|---|
| Reference images | 300 | **~3,300** (freshly extracted, **disjoint** from training) | full-rank covariance |
| Generated / model | 256 | **2,048** | full-rank covariance |
| Prompts | *"an impressionist painting of a landscape, in the style of Claude-Monet"* | **neutral bank** — 240 unique, **zero** style words | the old prompts *told* the model the answer; base looked strong for free |
| LoRA scale | 1.0 | **1.5** (tuned) + 1.0 (control) | we were under-applying our own adapter |
| Full fine-tune | 4 img/step (**half** of LoRA's 8) | **retrained at 8 img/step** | removes the images-seen confound |
| Floor check | — | **re-measure REAL-vs-REAL at the new N** | proves the ruler now has resolving power |

### 8.4 Process notes — what we hit while building it
- **Needed a `--skip` flag.** The WikiArt stream is deterministic and v1 already consumed the first
  1,500 Impressionism matches (300 held-out + 1,200 train). To get a *fresh, non-overlapping* reference
  we skip those first 1,500 matches. Using training images as an FID reference would have been a serious
  methodological error (the fine-tunes had memorised part of it).
- **🔴 A smoke test caught a design flaw before it cost 4 hours.** Naively neutralising the captions
  collapsed 1,200 captions to only **~11 unique prompts** ("a landscape", "a portrait", …). Generating
  2,048 images from 11 prompts produces a far *narrower* distribution than the real reference — which
  would inflate FID for **every** model and confound style with content. FID compares *distributions*, so
  the content must be broadly matched and only the style allowed to vary. **Fix:** a diverse, style-free
  **prompt bank** (30 Impressionist subjects × 8 neutral modifiers = **240 unique prompts**).
- **Memory check before committing.** Full-FT at batch 2 doubles the activation memory. We verified it
  fits (2.36 img/s, no OOM) with a 3-step probe *before* launching an 85-minute training run.
- **🔴 The reference download died to a network blip — and revealed a library flaw.** The first extraction
  crashed with a read-timeout, then `[Errno 11001] getaddrinfo failed` (a transient **DNS** failure). Worse,
  `huggingface_hub` then **closed its HTTP client**, so its own retry logic immediately failed with
  `RuntimeError: Cannot send a request, as the client has been closed` — the retries could never succeed.
  **Root problem:** `datasets` streaming has **no resume**, and we were asking it to stream ~17,000 rows
  over ~50 minutes — a long window for a single hiccup. **Fix:** a wrapper
  (`environment/extract_reference.ps1`) that retries in a **fresh process** (discarding the broken client)
  and uses `--skip` + a new `--name-offset` to **continue from whatever is already on disk**. Cached
  parquet shards make each resume fast. *(Lesson: any long unattended network job needs resumability, not
  just retries.)*

- **🔴 A host-session restart killed both background jobs mid-flight** (the matched full-FT at ~step
  500/1500; the reference extraction between retries). Two decisions: **(a)** discard the partial
  checkpoint and retrain fresh — the matched run exists *solely* to be a clean comparison, and resuming
  without optimizer state would put an asterisk on the headline claim to save ~27 minutes; **(b)** relaunch
  both jobs **detached under Windows Task Scheduler**, so multi-hour unattended work no longer depends on
  the interactive session staying alive. *(Third resilience lesson of the project, after
  checkpoint/resume for training and resume-by-skip for network streams.)*

- **🔴 The sweep wedged on VRAM oversubscription (WDDM).** With the desktop heavily loaded in the
  evening (browsers, Steam, Discord, Docker, photo viewers — ~10 GB of the shared 32 GB), our fp32
  batch-16 generation crossed the memory ceiling: Windows' WDDM driver began paging GPU memory,
  throughput collapsed from **0.9 s/img to 62 s/img (~65×)**, and the process finally hung — cold GPU
  (34 °C), 15 % util, 31.8/32.6 GB allocated, 26 minutes without progress. The failure was *silent*: no
  exception, no exit code — our first watcher only covered crashes and thus timed out uselessly.
  **Fixes:** kill + resume with generation **batch 16→8** (halves VRAM; provably does not change the
  generated images, so the comparison stays valid), and a new watcher that also alarms on **log
  silence > 15 min** (the wedge signature). *(Fourth resilience lesson: on a desktop GPU the OS is a
  co-tenant — leave VRAM headroom, and monitor for stalls, not just failures.)*

- **🔴 A midnight reboot killed the resumed sweep** (machine boot at 00:35:46, mid-generation at
  1056/2048). A no-trigger scheduled task does not restart after boot; a final-leg script finished the
  remaining work next day. *(Fifth resilience lesson: for multi-day unattended work, add a boot trigger
  — or expect to relaunch.)* In total this evaluation survived **a network-killed download, a VRAM
  wedge, and a reboot.*

### 8.5 Results — the corrected evaluation (final)

Reference: **2,800 real held-out paintings** (disjoint from training). Per model: **2,048 images**
from **240 neutral prompts (zero style words)**. Full data: `experiments/eval_v2_results.json`.

**The ruler is fixed.** Real-vs-real floor at this sample size: **FID 37.6** (was 156.7 at v1's N —
a 119-point collapse of pure estimator bias). Differences between models are now meaningful.

| Model | FID ↓ | above floor | CLIP ↑ |
|---|---|---|---|
| Base SD-1.5 (no fine-tune) | 128.3 | +90.7 | 32.72 |
| LoRA r4 @×1.5 | 116.9 | +79.3 | 32.67 |
| LoRA r16 @×1.0 | 119.3 | +81.7 | 32.89 |
| **LoRA r16 @×1.5** | **112.8** | **+75.2** | **32.93** |
| LoRA r64 @×1.5 *(added post-hoc)* | 114.5 | +76.9 | **33.10** |
| DreamBooth @×1.5, no `sks` *(added post-hoc)* | 119.7 | +82.1 | 32.71 |
| DreamBooth @×1.5, **+ `", in sks style"`** *(added post-hoc)* | 119.7 | +82.1 | **31.66** |
| Full FT (4 img/step, v1 confounded) | 123.0 | +85.4 | 32.78 |
| Full FT **matched** (8 img/step, fair) | 121.5 | +83.9 | 32.84 |

**What v2 established (and v1 could not):**
1. **Fine-tuning demonstrably works.** With *no style word in the prompt*, every fine-tune beats base;
   the best gap is **−15.5 FID**. Under v1's broken ruler this ordering was invisible.
2. **LoRA doesn't just match full fine-tuning — it beats it.** In the fair, images-seen-matched fight:
   LoRA r16@1.5 **112.8** vs matched full-FT **121.5** (−8.7). Even the **3.3 MB** rank-4 adapter
   (116.9) outscores the **3.44 GB** full fine-tune. The headline claim upgrades from *matches* to
   **beats at <0.4 % of the parameters**.
3. **The scale knob is worth real FID.** Same r16 weights: ×1.0 → ×1.5 = **−6.5 FID, for free** —
   quantifying the "we under-applied our own model" discovery.
4. **The rank story resolves.** At matched scale, r16 (112.8) beats r4 (116.9) — v1's "r4 nominal
   best" was, as suspected, estimator noise.
5. **The confound is closed.** Doubling full-FT's images-seen helped it only slightly (123.0 → 121.5);
   the verdict does not change. Running the matched control was still the right call — now nobody can
   attribute the LoRA win to unequal data.
6. **CLIP is nearly flat among the style models (32.67–33.10)** — with neutral *content* prompts they
   all depict the content, and the style signal lives in FID, as designed. Two structured effects do
   survive: rank tracks alignment (r4 32.67 < r16 32.93 < r64 33.10 — more capacity renders more of
   the prompt), and the `sks` trigger variant sits a full point below everything (31.66, point 10).
7. **Honesty note:** +75 above floor means our best model is still far from statistically
   indistinguishable from real Impressionism (expected at 1,200 training images), and v1 vs v2 numbers
   are **not comparable** (different prompts and N). Only within-v2 comparisons are valid.
8. **r64, vindicated — the narrative bowed to data a second time.** The two models v2 initially
   omitted were generated and scored post-hoc: **r64 @×1.5 took second place (114.5) with the highest
   CLIP (33.10)** — v1's "r64 is worst" was an artifact of scale 1.0 + the broken ruler. Its
   composition drift remains a real *qualitative* trait, but it barely costs distributional distance.
9. **DreamBooth generalized beyond its trigger:** 119.7 with **no `sks` token in any prompt** — the
   style leaked into general behaviour, beating both full fine-tunes. Final standing: **every adapter
   (3.3–51 MB) beats every full fine-tune (3.44 GB).**
10. **🆕 …and the trigger token turns out to be *inert* — a flaw v1 never even tested.** v1 scored
    DreamBooth only on prompts that never contain `sks`, i.e. with the method's entire mechanism
    switched off, and nobody noticed. So we ran it **both ways** on identical prompts:

    | | FID ↓ | CLIP ↑ |
    |---|---|---|
    | neutral prompt | 119.65 | 32.71 |
    | + `", in sks style"` | 119.66 | **31.66** |

    **FID is identical to 0.01 — the trigger contributes no style at all — while CLIP falls 1.05.**
    Mechanistically clear in hindsight: we bound `sks` using **1,200 instance images under a single
    fixed prompt**, so the token never received a contrastive signal separating it from the constant
    context. The adapter learned an **unconditional style shift**, and `sks` is dead weight that only
    drags the text embedding away from the content words. **This was style-LoRA training wearing a
    DreamBooth costume** — true few-shot DreamBooth binds a handful of instance images against a
    class prior, a different regime. Point 9's "generalized beyond its trigger" is therefore better
    stated as: *it never bound to the trigger in the first place.*

> **The meta-lesson, sharper than v1's.** Fixing the ruler did not merely sharpen our conclusions — it
> **reversed one** (rank 64 went from worst LoRA to second-best overall) and **exposed an experiment we
> had never actually run** (the trigger). Every quantitative claim in the v1 report was downstream of
> an estimator we had never validated. **Measure your measurement first.**

---

## 9. Evidence index

| Claim | Artifact |
|---|---|
| EMA noise → recovery | `outputs/phase1/p1a_butterflies/samples/ddim_0004000.png` → `ddim_ema_0008000.png` |
| EMA fix | `src/common/ema.py` (warmup), `src/phase1_ddpm_from_scratch/train.py` (`--reset-ema`) |
| lr-too-high failure | `outputs/phase1/p1_r01_lr2e3/`, loss ≈0.98 in `experiments/sweep.log` |
| From-scratch limits | `outputs/phase1/p1b_impressionism/samples/` |
| Method comparison | `report/figures/method_compare.png` (same prompt + seed, 6 models) |
| Learning curve | `report/figures/phase1a_loss.png` |
| All runs & timings | `experiments/sweep.log`, `experiments/sweep_summary.txt` |
| Metrics table | `experiments/RESULTS.md`, `report/report.md` §6 |
| Full analysis | `report/report.md` |
