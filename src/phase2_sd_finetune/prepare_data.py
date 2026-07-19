"""Phase-2 data prep: build an Impressionist subset from WikiArt.

Streams `huggan/wikiart`, keeps style == Impressionism, center-crops to --size, and
saves JPGs + a `metadata.jsonl` (HF imagefolder format) with templated captions for
text-conditioned fine-tuning. Reserves a held-out split as the FID reference.
The same 512px folder is reused by Phase 1b at 64px (the loader downsizes on the fly).

Example:
    python src/phase2_sd_finetune/prepare_data.py --max-images 2000 --holdout 400
Quick test:
    python src/phase2_sd_finetune/prepare_data.py --max-images 16 --holdout 4
"""
import argparse
import json
import pathlib

from torchvision.transforms import functional as TF
from datasets import load_dataset

ROOT = pathlib.Path(__file__).resolve().parents[2]


def caption_for(genre, artist):
    base = "an impressionist painting"
    g = genre.replace("_", " ") if genre else None
    a = artist.replace("_", " ").title() if artist else None
    if g and "unknown" not in g.lower():
        base += f" of a {g}"
    if a and "unknown" not in a.lower():
        base += f", in the style of {a}"
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--holdout", type=int, default=400)
    ap.add_argument("--style", default="Impressionism")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N matching images (so a second extraction does NOT "
                         "overlap an earlier one — e.g. --skip 1500 to get fresh held-out data)")
    ap.add_argument("--name-offset", type=int, default=0,
                    help="start held-out filenames at this index — lets a retry RESUME after a "
                         "network failure without overwriting what was already downloaded")
    args = ap.parse_args()

    out = pathlib.Path(args.out) if args.out else ROOT / "data" / f"impressionism_{args.size}"
    train_dir, held_dir = out / "train", out / "heldout"
    train_dir.mkdir(parents=True, exist_ok=True)
    held_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    feats = ds.features
    sidx = feats["style"].names.index(args.style)
    gnames, anames = feats["genre"].names, feats["artist"].names
    print(f"target style {args.style!r} = index {sidx}; writing to {out}", flush=True)

    meta = open(train_dir / "metadata.jsonl", "w", encoding="utf-8")
    n = nh = scanned = skipped = 0
    for ex in ds:
        scanned += 1
        if ex["style"] != sidx:
            continue
        if skipped < args.skip:          # already consumed by an earlier extraction -> keep sets disjoint
            skipped += 1
            continue
        try:
            img = ex["image"].convert("RGB")
        except Exception:
            continue
        img = TF.center_crop(TF.resize(img, args.size), [args.size, args.size])
        genre = gnames[ex["genre"]] if ex["genre"] is not None else None
        artist = anames[ex["artist"]] if ex["artist"] is not None else None
        if nh < args.holdout:
            img.save(held_dir / f"imp_{args.name_offset + nh:05d}.jpg", quality=95)
            nh += 1
        else:
            fn = f"imp_{n:05d}.jpg"
            img.save(train_dir / fn, quality=95)
            meta.write(json.dumps({"file_name": fn, "text": caption_for(genre, artist)}) + "\n")
            n += 1
        if (n + nh) % 50 == 0:
            meta.flush()
            print(f"  scanned={scanned} skipped={skipped} kept train={n} heldout={nh}", flush=True)
        if nh >= args.holdout and n >= args.max_images:   # allows --max-images 0 (held-out only)
            break
    meta.close()
    print(f"DONE: scanned={scanned} train={n} heldout={nh} -> {out}", flush=True)


if __name__ == "__main__":
    main()
