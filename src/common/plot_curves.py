"""Plot a scalar (e.g. training loss) from a TensorBoard logdir to a PNG (report figures).

Usage:
  python src/common/plot_curves.py --logdir outputs/phase1/p1a_butterflies/tb \
         --tag loss --out report/figures/phase1a_loss.png --smooth 3
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--tag", default="loss")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Training loss")
    ap.add_argument("--smooth", type=int, default=1)
    args = ap.parse_args()

    ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if args.tag not in ea.Tags().get("scalars", []):
        raise SystemExit(f"tag {args.tag!r} not found. Available: {ea.Tags().get('scalars')}")
    events = ea.Scalars(args.tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])

    plt.figure(figsize=(7, 4))
    plt.plot(steps, vals, alpha=0.3, label="raw")
    if args.smooth > 1 and len(vals) >= args.smooth:
        k = args.smooth
        sm = np.convolve(vals, np.ones(k) / k, mode="valid")
        plt.plot(steps[k - 1:], sm, label=f"smoothed (k={k})")
    plt.xlabel("step")
    plt.ylabel(args.tag)
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out} | {len(steps)} points, last={vals[-1]:.4f}")


if __name__ == "__main__":
    main()
