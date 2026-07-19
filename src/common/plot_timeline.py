"""Generate the project journey timeline figure (grounded in real build timestamps).

Times are hours elapsed from the first artifact (18:53) through assembly (~03:15),
derived from file mtimes and experiments/sweep.log.

    python src/common/plot_timeline.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

C = {
    "setup": "#6b7280",  # grey
    "p1":    "#2563eb",  # blue
    "fix":   "#16a34a",  # green (a fix, highlighted)
    "p2":    "#7c3aed",  # violet
    "sweep": "#0d9488",  # teal
    "asm":   "#b45309",  # amber
}

# (label, start_h, duration_h, colour_key)  — one task per row, chronological
TASKS = [
    ("env install (torch cu128) + related-work review", 0.00, 0.25, "setup"),
    ("smoke test (16 px, fake data)",                   0.20, 0.10, "p1"),
    ("butterflies v1 + throughput probes",              0.30, 0.30, "p1"),
    ("butterflies training (from scratch)",             0.60, 0.63, "p1"),
    ("EMA fix → resume → recovered",                    1.23, 0.35, "fix"),
    ("WikiArt data prep + Phase-2 trainers written",    0.85, 0.90, "p2"),
    ("LoRA r16 — first real Impressionist fine-tune",   1.68, 1.02, "p2"),
    ("sweep · base  (infer + eval)",                    2.88, 0.10, "sweep"),
    ("sweep · LoRA r4  → train + eval",                 2.98, 1.10, "sweep"),
    ("sweep · LoRA r64 → train + eval",                 4.08, 1.12, "sweep"),
    ("sweep · Full fine-tune → train + eval",           5.20, 0.98, "sweep"),
    ("sweep · DreamBooth → train + eval",               6.18, 1.63, "sweep"),
    ("sweep · Phase-1 extras (lr-fail, Imp-64)",        7.82, 0.23, "sweep"),
    ("assembly · re-infer r16 · figures · corrections", 8.05, 0.35, "asm"),
]

# (time_h, text, kind, stagger_level)   kind: fail | fix
EVENTS = [
    (0.45, "GPU-resident data → no speed-up:\nwe are COMPUTE-bound (stop optimising)", "fail", 2),
    (1.05, "EMA samples = PURE NOISE", "fail", 0),
    (1.30, "EMA warmup + reset\n→ recovered", "fix", 1),
    (2.98, "infer crash (Windows file lock)\n→ hardened mid-sweep", "fail", 2),
    (8.03, "p1b native crash\n→ re-ran OK", "fail", 0),
    (8.38, "overclaim retracted\n(the data won)", "fix", 1),
]

FAIL, FIXC = "#dc2626", "#16a34a"


def main():
    n = len(TASKS)
    fig, ax = plt.subplots(figsize=(16, 8.2))

    # --- task bars (row 0 at top) ---
    for i, (label, start, dur, key) in enumerate(TASKS):
        y = n - 1 - i
        ax.barh(y, dur, left=start, height=0.62, color=C[key], alpha=0.95,
                edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(start + dur + 0.08, y, f"{dur*60:.0f}m", va="center", ha="left",
                fontsize=7.5, color="#4b5563", zorder=3)

    # --- staggered event call-outs above the chart ---
    levels = {0: n + 2.35, 1: n + 1.35, 2: n + 0.35}
    for t, text, kind, lvl in EVENTS:
        col = FAIL if kind == "fail" else FIXC
        ax.plot([t, t], [-0.75, levels[lvl] - 0.12], color=col, ls=":", lw=1.2, alpha=0.6, zorder=1)
        ax.plot(t, n - 0.35, marker="v" if kind == "fail" else "^", color=col,
                markersize=8, zorder=5, clip_on=False)
        ax.text(t, levels[lvl], text, ha="center", va="center", color=col, fontsize=8.4,
                fontweight="bold", linespacing=1.35, zorder=5,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=col, lw=0.9, alpha=0.95))

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([t[0] for t in TASKS], fontsize=9.5)
    ax.set_ylim(-0.9, n + 3.1)
    ax.set_xlim(-0.1, 9.1)

    ticks = list(range(0, 9))
    clock = ["18:53", "19:53", "20:53", "21:53", "22:53", "23:53", "00:53", "01:53", "02:53"]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{c}\n(+{h}h)" for h, c in zip(ticks, clock)], fontsize=8.5)
    ax.set_xlabel("Elapsed time  —  total build ≈ 8 h 15 m   (overnight sweep = 5 h 10 m)",
                  fontsize=10.5, labelpad=8)
    ax.set_title("Project Journey — what we built, what broke, and how we fixed it",
                 fontsize=14, fontweight="bold", pad=18)
    ax.grid(axis="x", alpha=0.22, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=C["setup"], label="setup"),
               Patch(facecolor=C["p1"], label="Phase 1 (from scratch)"),
               Patch(facecolor=C["p2"], label="Phase 2 (Stable Diffusion)"),
               Patch(facecolor=C["sweep"], label="overnight sweep"),
               Patch(facecolor=C["asm"], label="assembly"),
               Patch(facecolor=FAIL, label="✗ failure discovered"),
               Patch(facecolor=FIXC, label="✓ fix applied")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=7, fontsize=8.6, frameon=False)

    out = ROOT / "report" / "figures" / "journey_timeline.png"
    plt.tight_layout()
    plt.savefig(out, dpi=145, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
