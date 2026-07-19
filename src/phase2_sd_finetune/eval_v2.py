"""EVALUATION v2 — FID done properly.

Why v2 exists (see JOURNEY.md): our v1 FID used 256 generated vs 300 real images. FID fits a
**2048-dimensional** Gaussian, so a covariance estimated from 256 samples is rank-deficient and the
score is dominated by bias. We proved it: two sets of *genuine* Impressionist paintings scored
FID = 156.7 against each other — i.e. the metric could not even distinguish real art from real art,
let alone rank our six models (which all landed at 159-167).

v2 fixes the measurement:
  * reference : ~3,300 held-out real paintings (disjoint from training data)
  * generated : 2,048 images per model (full-rank covariance)
  * prompts   : NEUTRAL (no "impressionist", no artist) — isolates what the fine-tune adds
  * plus a re-measured REAL-vs-REAL floor at the new N, to prove the ruler now has resolving power

    python src/phase2_sd_finetune/eval_v2.py --models base=outputs/phase2_eval_v2/base/eval_samples ...
"""
import argparse
import copy
import json
import pathlib

import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png"}


def paths_in(folder):
    return sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def feed(metric, paths, real, device, bs=32, size=299):
    """Stream images from disk into the FID metric (avoids holding thousands in RAM)."""
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        t = torch.stack([TF.pil_to_tensor(Image.open(p).convert("RGB").resize((size, size)))
                         for p in chunk])
        metric.update((t.float() / 255.0).to(device), real=real)


def clip_score(gen_paths, captions_path, device, model_name="openai/clip-vit-base-patch16", bs=16):
    if not pathlib.Path(captions_path).exists():
        return None
    caps = {}
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            caps[r["file_name"]] = r["text"]
    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    sims = []
    with torch.no_grad():
        for i in range(0, len(gen_paths), bs):
            chunk = gen_paths[i:i + bs]
            pil = [Image.open(p).convert("RGB") for p in chunk]
            txt = [caps.get(p.name, "a painting") for p in chunk]
            inp = proc(text=txt, images=pil, return_tensors="pt", padding=True, truncation=True).to(device)
            out = cm(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims.append((ie * te).sum(-1).cpu())
    return float(torch.cat(sims).mean()) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dirs", nargs="+", required=True, help="folders of REAL held-out paintings")
    ap.add_argument("--models", nargs="+", required=True, help="name=path/to/eval_samples ...")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "eval_v2_results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torchmetrics.image.fid import FrechetInceptionDistance

    ref_paths = []
    for d in args.ref_dirs:
        ref_paths += paths_in(d)
    print(f"REFERENCE: {len(ref_paths)} real held-out paintings (from {len(args.ref_dirs)} folder(s))")
    if len(ref_paths) < 2048:
        print(f"  ⚠ warning: reference N={len(ref_paths)} < 2048 → covariance still rank-deficient")

    # ---------- the metric's noise floor at THIS N (real vs real) ----------
    half = len(ref_paths) // 2
    print(f"\n[floor] REAL vs REAL  ({half} vs {half}) — a perfect model cannot beat this ...")
    f = FrechetInceptionDistance(normalize=True).to(device)
    feed(f, ref_paths[:half], True, device)
    feed(f, ref_paths[half:2 * half], False, device)
    floor = float(f.compute())
    print(f"[floor] FID = {floor:.2f}\n")
    del f
    torch.cuda.empty_cache()

    # ---------- reference features computed ONCE, reused for every model ----------
    print("computing reference Inception statistics once ...")
    base_metric = FrechetInceptionDistance(normalize=True).to(device)
    feed(base_metric, ref_paths, True, device)

    results = {"reference_n": len(ref_paths), "real_vs_real_floor": floor, "models": {}}
    for spec in args.models:
        name, gdir = spec.split("=", 1)
        gpaths = paths_in(gdir)
        m = copy.deepcopy(base_metric)
        feed(m, gpaths, False, device)
        fid = float(m.compute())
        del m
        torch.cuda.empty_cache()

        cs = None
        if not args.no_clip:
            cs = clip_score(gpaths, str(pathlib.Path(gdir) / "prompts.jsonl"), device)

        results["models"][name] = {"n": len(gpaths), "fid": fid, "clip": cs,
                                   "fid_above_floor": fid - floor}
        cstr = f"CLIP {cs:5.2f}" if cs is not None else "CLIP   n/a"
        print(f"  {name:<18} N={len(gpaths):5d}  FID {fid:7.2f}  ({fid-floor:+6.2f} vs floor)  {cstr}")

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")
    print("\nFID minus floor is the honest number: how far the model sits ABOVE what real art scores.")


if __name__ == "__main__":
    main()
