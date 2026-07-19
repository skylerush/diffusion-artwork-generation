"""Learning curves for every training run in the project, as one two-panel figure.

Why not "accuracy" or "AUC": both are CLASSIFICATION metrics — they compare a prediction against a
known-correct label. A diffusion model synthesises an image from noise, so no per-prompt ground
truth exists and neither metric is defined. The learning curve here is the epsilon-prediction MSE,
whose floor-for-random is exactly 1.0 (the noise is drawn from N(0,1) with unit variance). So
(1 - loss) reads directly as "fraction of the noise variance explained".

Phase-1 losses come from TensorBoard; Phase-2 never wrote TensorBoard, so its curves are parsed out
of the stdout captured in experiments/*.log.

    python src/common/plot_loss_curves.py
"""
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

P1_STYLE = {
    "p1a_butterflies":   ("#2563eb", "p1a butterflies (lr 2e-4)"),
    "p1b_impressionism": ("#7c3aed", "p1b impressionism (lr 2e-4)"),
    "p1_r01_lr2e3":      ("#dc2626", "p1-r01 lr 2e-3  (deliberate failure)"),
}
P2_STYLE = {
    "lora_r4":         ("#16a34a", "LoRA r4"),
    "lora_r64":        ("#b45309", "LoRA r64"),
    "dreambooth":      ("#db2777", "DreamBooth"),
    "full_ft":         ("#0d9488", "Full FT (4 img/step)"),
    "full_ft_matched": ("#1e40af", "Full FT matched (8 img/step)"),
}
STEP_LINE = re.compile(r"^step\s+(\d+)\s*\|\s*loss\s+([0-9.]+)")
BEGIN_TRAIN = re.compile(r"BEGIN train:(\S+)")


def dedupe(steps, losses):
    """Collapse overlapping step ranges, keeping the LATEST value logged for each step.

    Two runs in this project produce non-monotonic step sequences and would otherwise draw a line
    doubling back on itself:
      * p1a_butterflies was RESUMED at step 5000, so its TensorBoard dir holds both sessions.
      * full_ft_matched's first attempt died at ~step 560 and was restarted from zero, so the log
        contains two overlapping sequences.
    Processing in chronological order and letting later entries win keeps the run that survived.
    """
    seen = {}
    for s, l in zip(steps, losses):
        seen[int(s)] = float(l)
    ks = np.array(sorted(seen))
    return ks, np.array([seen[k] for k in ks])


def phase1_curves():
    """{run: (steps, losses)} from each run's TensorBoard scalars."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for run in P1_STYLE:
        tb = ROOT / "outputs" / "phase1" / run / "tb"
        if not tb.exists():
            continue
        ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})
        ea.Reload()
        if "loss" not in ea.Tags().get("scalars", []):
            continue
        ev = ea.Scalars("loss")
        out[run] = dedupe([e.step for e in ev], [e.value for e in ev])
    return out


def phase2_curves():
    """{run: (steps, losses)} parsed from the training stdout captured in the sweep logs."""
    out = {}
    for log in ("sweep.log", "eval_v2.log", "eval_v2_ext.log"):
        p = ROOT / "experiments" / log
        if not p.exists():
            continue
        current = None
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = BEGIN_TRAIN.search(line)
            if m:
                current = m.group(1)
                continue
            m = STEP_LINE.match(line.strip())
            if m and current in P2_STYLE:
                out.setdefault(current, ([], []))
                out[current][0].append(int(m.group(1)))
                out[current][1].append(float(m.group(2)))
    return {k: dedupe(v[0], v[1]) for k, v in out.items() if len(v[0]) > 2}


def smooth(y, k=5):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main():
    p1, p2 = phase1_curves(), phase2_curves()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

    # ---- Panel A: from-scratch DDPM (log scale spans 0.01 -> 1.0) ----
    for run, (steps, loss) in p1.items():
        c, label = P1_STYLE[run]
        ax1.plot(steps, loss, color=c, lw=1.6, label=f"{label} — final {loss[-1]:.4f}")
    ax1.axhline(1.0, color="#6b7280", ls="--", lw=1.2)
    ax1.text(ax1.get_xlim()[1] * 0.98, 1.02, "loss = 1.0: predicting nothing (noise has unit variance)",
             ha="right", va="bottom", fontsize=8.5, color="#6b7280")
    ax1.set_yscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("ε-prediction MSE  (log scale)")
    ax1.set_title("Phase 1 — DDPM from scratch, 64px", fontweight="bold")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8.5, loc="lower left")

    # ---- Panel B: SD-1.5 fine-tunes (all start from a trained prior, so no descent to ~0) ----
    for run, (steps, loss) in sorted(p2.items()):
        c, label = P2_STYLE[run]
        ax2.plot(steps, loss, color=c, lw=0.7, alpha=0.25)
        sm = smooth(loss)
        ax2.plot(steps[len(steps) - len(sm):], sm, color=c, lw=1.8, label=label)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("ε-prediction MSE (latent space)")
    ax2.set_title("Phase 2 — Stable Diffusion fine-tunes, 512px", fontweight="bold")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8.5)
    ax2.text(0.98, 0.03,
             "DreamBooth sits higher because its loss is the SUM of two MSE terms\n"
             "(instance + prior preservation) — not because it fits worse.\n"
             "Phase-1 and Phase-2 losses are NOT comparable: different spaces\n"
             "(pixel vs latent) and exposure (~1000 vs 10 epochs).",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8, color="#4b5563",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f9fafb", ec="#d1d5db", lw=0.8))

    fig.suptitle("Learning curves — every training run", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    out = ROOT / "report" / "figures" / "loss_curves_all.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    for name, (s, l) in list(p1.items()) + sorted(p2.items()):
        print(f"  {name:<20} {len(s):>4} points, final {l[-1]:.4f}, min {l.min():.4f}")
    missing = set(P2_STYLE) - set(p2)
    if missing:
        print(f"  (no logged curve for: {', '.join(sorted(missing))})")
    print("  note: lora_r16 was trained interactively before the sweep; its stdout was not captured.")


if __name__ == "__main__":
    sys.exit(main())
