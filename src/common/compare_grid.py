"""Tile labeled images into a single grid for report comparison figures.

Each positional arg is 'Label::path/to/image'. Useful for the method comparison
(Base vs LoRA vs Full-FT vs DreamBooth on the same prompt) and rank-sweep figures.

Usage:
  python src/common/compare_grid.py --cols 4 --out report/figures/method_compare.png \
      "Base::outputs/phase2/base/eval_samples/gen_00000.jpg" \
      "LoRA r16::outputs/phase2/lora_r16/eval_samples/gen_00000.jpg" \
      "Full-FT::outputs/phase2/full_ft/eval_samples/gen_00000.jpg" \
      "DreamBooth::outputs/phase2/dreambooth/eval_samples/gen_00000.jpg"
"""
import argparse

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="'Label::imagepath'")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = [it.split("::", 1) for it in args.items]
    cols = max(1, args.cols)
    rows = (len(pairs) + cols - 1) // cols
    cell, pad = args.cell, 22
    canvas = Image.new("RGB", (cols * cell, rows * (cell + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(pairs):
        r, c = divmod(i, cols)
        img = Image.open(path).convert("RGB").resize((cell, cell))
        x, y = c * cell, r * (cell + pad) + pad
        canvas.paste(img, (x, y))
        draw.text((x + 4, y - 16), label, fill="black")
    canvas.save(args.out)
    print(f"saved {args.out} ({len(pairs)} tiles, {cols}x{rows})")


if __name__ == "__main__":
    main()
