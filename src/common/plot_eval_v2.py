"""Bar chart of the final (v2) evaluation from experiments/eval_v2_results.json.

    python src/common/plot_eval_v2.py   ->  report/figures/eval_v2_results.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

LABELS = {
    "base":            ("Base SD-1.5\n(no fine-tune)",        "#6b7280"),
    "lora_r4":         ("LoRA r4 @1.5\n(3.3 MB)",             "#a78bfa"),
    "lora_r16_s1.0":   ("LoRA r16 @1.0\n(12.8 MB)",           "#8b5cf6"),
    "lora_r16_s1.5":   ("LoRA r16 @1.5\n(12.8 MB)",           "#6d28d9"),
    "lora_r64":        ("LoRA r64 @1.5\n(51 MB)",             "#4c1d95"),
    "dreambooth":      ("DreamBooth @1.5\n(12.8 MB)",         "#0d9488"),
    "full_ft":         ("Full FT 4img/step\n(3.44 GB)",       "#f59e0b"),
    "full_ft_matched": ("Full FT matched\n8img/step (3.44 GB)","#b45309"),
}


def main():
    data = json.loads((ROOT / "experiments" / "eval_v2_results.json").read_text())
    floor = data["real_vs_real_floor"]
    items = sorted(data["models"].items(), key=lambda kv: kv[1]["fid"])

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    names = [LABELS[k][0] for k, _ in items]
    fids = [v["fid"] for _, v in items]
    cols = [LABELS[k][1] for k, _ in items]

    bars = ax.bar(names, fids, color=cols, width=0.62, zorder=3)
    for b, (_, v) in zip(bars, items):
        ax.text(b.get_x() + b.get_width() / 2, v["fid"] + 1.2,
                f"{v['fid']:.1f}\n(+{v['fid_above_floor']:.1f})",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", zorder=4)
    bars[0].set_edgecolor("#1f2937"); bars[0].set_linewidth(2.2)

    ax.axhline(floor, color="#16a34a", lw=2, ls="--", zorder=2)
    ax.text(len(items) - 0.42, floor + 1.3,
            f"real-vs-real floor = {floor:.1f}\n(a perfect model scores here)",
            ha="right", color="#16a34a", fontsize=9.5, fontweight="bold")

    ax.set_ylabel("FID vs 2,800 real held-out paintings   (lower = better)")
    ax.set_ylim(0, max(fids) * 1.16)
    ax.set_title("Evaluation v2 — honest ruler: 2,048 imgs/model · neutral prompts (zero style words)",
                 fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=9.5)

    out = ROOT / "report" / "figures" / "eval_v2_results.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
