"""Phase-2 evaluation: FID (vs held-out Impressionism) + CLIP score.

FID measures distributional closeness to real Impressionism; CLIP score measures how well
generated images match their prompts. Note: with a few hundred images FID is approximate —
we report it as a *relative* comparison across methods, not an absolute number.

CLIP score is computed directly from the CLIP model (cosine of normalized image/text
embeddings ×100) because torchmetrics' CLIPScore is incompatible with transformers 5.x.

Usage:
  python src/phase2_sd_finetune/eval.py --gen-dir outputs/phase2/lora_r16/eval_samples \
         --captions outputs/phase2/lora_r16/eval_samples/prompts.jsonl
"""
import argparse
import json
import pathlib

import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    return sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def load_uint8(paths, size=299):
    imgs = [TF.pil_to_tensor(Image.open(p).convert("RGB").resize((size, size))) for p in paths]
    return torch.stack(imgs) if imgs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--ref-dir", default=str(ROOT / "data" / "impressionism_512" / "heldout"))
    ap.add_argument("--captions", default=None, help="jsonl {file_name,text} for CLIP score")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_paths = list_images(args.gen_dir)
    ref_paths = list_images(args.ref_dir)
    if not gen_paths or not ref_paths:
        raise SystemExit(f"need images in both --gen-dir ({len(gen_paths)}) and --ref-dir ({len(ref_paths)})")
    gen = load_uint8(gen_paths)
    ref = load_uint8(ref_paths)
    print(f"gen={gen.shape[0]} images | ref={ref.shape[0]} images", flush=True)

    # ---- FID ----
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(normalize=True).to(device)
    for imgs, real in ((ref, True), (gen, False)):
        for i in range(0, imgs.shape[0], 32):
            fid.update((imgs[i:i + 32].float() / 255.0).to(device), real=real)
    print(f"FID = {float(fid.compute()):.2f}  (lower is better)", flush=True)

    # ---- CLIP score (manual, transformers-5.x safe) ----
    if args.captions:
        caps = {}
        with open(args.captions, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                caps[r["file_name"]] = r["text"]
        texts = [caps.get(p.name, "an impressionist painting") for p in gen_paths]

        from transformers import CLIPModel, CLIPProcessor
        cm = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
        proc = CLIPProcessor.from_pretrained(args.clip_model)
        sims = []
        with torch.no_grad():
            for i in range(0, len(gen_paths), 16):
                pil = [Image.open(p).convert("RGB") for p in gen_paths[i:i + 16]]
                inp = proc(text=texts[i:i + 16], images=pil, return_tensors="pt",
                           padding=True, truncation=True).to(device)
                out = cm(**inp)
                ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
                sims.append((ie * te).sum(-1).cpu())
        clip_score = float(torch.cat(sims).mean()) * 100.0
        print(f"CLIPScore = {clip_score:.2f}  (higher is better)", flush=True)


if __name__ == "__main__":
    main()
