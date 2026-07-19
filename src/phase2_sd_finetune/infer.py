"""Generate a batch of images from a fine-tuned SD checkpoint (LoRA or full UNet) for evaluation.

Produces N images into --out (prompts drawn from the training captions) plus a prompts.jsonl,
so eval.py can compute FID (vs held-out) and CLIP score. Also supports the BASE model as a
baseline (no checkpoint) for the comparison.

Usage:
  python src/phase2_sd_finetune/infer.py --lora outputs/phase2/lora_r16/ckpt/lora_last.pt --rank 16 --run-name lora_r16 --n 300
  python src/phase2_sd_finetune/infer.py --unet outputs/phase2/full_ft/ckpt/unet_last --run-name full_ft --n 300
  python src/phase2_sd_finetune/infer.py --run-name base --n 300     # baseline (no fine-tune)
"""
import argparse
import itertools
import json
import pathlib
import re
import sys
import time

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import DEFAULT_BASE  # noqa: E402


def load_prompts(captions_path, n):
    with open(captions_path, encoding="utf-8") as f:
        caps = [json.loads(line)["text"] for line in f]
    return list(itertools.islice(itertools.cycle(caps), n))


# ---- neutral prompts: NO style cues at all --------------------------------------------------
# Our captions were "an impressionist painting of a landscape, in the style of Claude-Monet".
# BOTH "impressionist painting" and the artist name cue the style, so the base model looked strong
# for free. Here we drop every style cue and keep only *content*.
#
# Why a prompt BANK and not just the caption genres: neutralising the captions collapses 1,200 of
# them to ~11 unique strings ("a landscape", "a portrait", ...). Generating 2,048 images from 11
# prompts yields a far narrower distribution than the real reference set, which inflates FID for
# every model. FID compares distributions, so the CONTENT must be broadly matched and only the
# STYLE allowed to differ. 30 subjects x 8 modifiers = 240 unique, style-free prompts.
_NEUTRAL_SUBJECTS = [
    "a river with tall trees", "a harbour with sailing boats", "a garden full of flowers",
    "a city boulevard with people", "a cathedral facade", "haystacks in a field",
    "water lilies on a pond", "a woman holding a parasol", "dancers on a stage",
    "a cafe terrace", "a stone bridge over a river", "a snow-covered village street",
    "a field of poppies", "a train station", "a beach with cliffs", "an orchard in bloom",
    "a woman reading by a window", "a still life with fruit", "a path through the woods",
    "a wheat field", "a canal lined with houses", "a park in spring",
    "a portrait of a young woman", "rowing boats on a lake", "a windmill on a hill",
    "a market square", "a pond with reeds", "a country road", "a courtyard garden",
    "a valley with distant mountains",
]
_NEUTRAL_MODIFIERS = [
    "", " at sunrise", " at sunset", " in summer", " in winter",
    " in the rain", " on a foggy morning", " under a cloudy sky",
]


def neutral_bank():
    return [s + m for s in _NEUTRAL_SUBJECTS for m in _NEUTRAL_MODIFIERS]


def set_lora_scale(unet, scale):
    """LoRA contributes `scale * B@A`; peft keeps `scale` in layer.scaling (default alpha/r)."""
    n = 0
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and sc:
            for k in sc:
                sc[k] = scale
            n += 1
    return n


def save_robust(img, path, tries=5):
    """Save with retry — the indexed Desktop can transiently lock a freshly written file."""
    for _ in range(tries - 1):
        try:
            img.save(path, quality=95)
            return
        except PermissionError:
            time.sleep(0.5)
    img.save(path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--lora", default=None, help="path to a LoRA state dict (.pt)")
    ap.add_argument("--unet", default=None, help="path to a saved full-FT unet dir")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (must match the checkpoint)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--captions", default=str(ROOT / "data" / "impressionism_512" / "train" / "metadata.jsonl"))
    ap.add_argument("--run-name", default="eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="strength of the LoRA delta at inference (free knob; 1.5-2.0 = stronger style)")
    ap.add_argument("--neutral", action="store_true",
                    help="strip ALL style cues from prompts (no 'impressionist', no artist) so the "
                         "measurement isolates what the fine-tuning itself contributes")
    ap.add_argument("--trigger-suffix", default=None,
                    help="append a trained trigger token to every prompt (DreamBooth). v1 never did "
                         "this — it scored DreamBooth on prompts that omit 'sks', i.e. with the "
                         "method's whole mechanism switched off. Keep the suffix free of English "
                         "style words (', in sks style', NOT ', in sks impressionist style') so the "
                         "comparison stays apples-to-apples with the neutral bank.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(args.base, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.unet:
        pipe.unet = UNet2DConditionModel.from_pretrained(args.unet)
        tag = "full-FT unet"
    elif args.lora:
        pipe.unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                         target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        set_peft_model_state_dict(pipe.unet, torch.load(args.lora, map_location="cpu"))
        tag = f"LoRA rank {args.rank}"
        if args.lora_scale != 1.0:
            k = set_lora_scale(pipe.unet, args.lora_scale)
            tag += f" @ scale {args.lora_scale} ({k} layers)"
    else:
        tag = "BASE (no fine-tune)"
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"model: {tag} | generating {args.n} images", flush=True)

    out = pathlib.Path(args.out) if args.out else ROOT / "outputs" / "phase2" / args.run_name / "eval_samples"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("gen_*.jpg"):  # clear stale images to avoid overwrite-lock issues
        try:
            old.unlink()
        except OSError:
            pass
    if args.neutral:
        bank = neutral_bank()
        prompts = list(itertools.islice(itertools.cycle(bank), args.n))
        print(f"NEUTRAL prompt bank: {len(bank)} unique, zero style words. e.g. {bank[:3]}", flush=True)
    else:
        prompts = load_prompts(args.captions, args.n)
    if args.trigger_suffix:
        prompts = [p + args.trigger_suffix for p in prompts]
        print(f"TRIGGER suffix {args.trigger_suffix!r} -> e.g. {prompts[0]!r}", flush=True)
    g = torch.Generator(device=device).manual_seed(args.seed)
    k = 0
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as meta:
        for i in range(0, args.n, args.batch):
            bp = prompts[i:i + args.batch]
            imgs = pipe(bp, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g).images
            for img, p in zip(imgs, bp):
                fn = f"gen_{k:05d}.jpg"
                save_robust(img, out / fn)
                meta.write(json.dumps({"file_name": fn, "text": p}) + "\n")
                k += 1
            print(f"  generated {k}/{args.n}", flush=True)
    print(f"INFER_DONE {k} images -> {out}", flush=True)


if __name__ == "__main__":
    main()
