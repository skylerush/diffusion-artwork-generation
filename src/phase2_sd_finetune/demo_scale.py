"""Does a STRONGER LoRA scale convert the landscapes too?

A LoRA's contribution is `scale · B@A`. That `scale` is a **free knob at inference time** — no
retraining needed (peft stores it as `layer.scaling[adapter]`, default = lora_alpha / r = 1.0).

We sweep it over the same prompts and the same seeds, with **no style word in the prompt**, so any
change is caused purely by turning our learned weights up or down.

    python src/phase2_sd_finetune/demo_scale.py
"""
import argparse
import pathlib
import sys

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import DEFAULT_BASE   # noqa: E402
from src.phase2_sd_finetune.demo import PROMPTS, _font       # noqa: E402

SCALES = [0.0, 1.0, 1.5, 2.0, 2.5]   # 0.0 == base model (LoRA delta zeroed out)


def set_lora_scale(unet, scale):
    """Set the LoRA scaling factor on every adapted layer (peft stores it in `.scaling`)."""
    n = 0
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and sc:
            for k in sc:
                sc[k] = scale
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=str(ROOT / "outputs" / "phase2" / "lora_r16" / "ckpt" / "lora_last.pt"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--out", default=str(ROOT / "report" / "figures" / "demo_lora_scale.png"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading SD-1.5 + our LoRA ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(DEFAULT_BASE, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                     target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    set_peft_model_state_dict(pipe.unet, torch.load(args.lora, map_location="cpu"))

    rows = []
    for s in SCALES:
        touched = set_lora_scale(pipe.unet, s)
        tag = "base (LoRA off)" if s == 0.0 else f"LoRA × {s}"
        print(f"\n[scale {s}] {tag}  ({touched} adapted layers)", flush=True)
        imgs = []
        for i, p in enumerate(PROMPTS):
            g = torch.Generator(device=device).manual_seed(args.seed + i)
            imgs.append(pipe(p, num_inference_steps=args.steps, guidance_scale=args.guidance,
                             generator=g).images[0])
            print(f"    [{i+1}/{len(PROMPTS)}] {p}", flush=True)
        rows.append((tag, imgs))

    # ---- labelled grid: one row per scale ----
    c, pad_l, pad_t, gap = args.cell, 175, 74, 10
    W = pad_l + c * len(PROMPTS)
    H = pad_t + len(rows) * (c + gap)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((14, 12), "LoRA strength sweep — same prompt, same seed, NO style word",
           fill="black", font=_font(19, bold=True))
    d.text((14, 38), "Turning our 12 MB adapter up: how far does it push SD toward Impressionism?",
           fill="#4b5563", font=_font(13))
    for i, p in enumerate(PROMPTS):
        d.text((pad_l + i * c + 6, pad_t - 18), f'"{p[:40]}"', fill="#374151", font=_font(11))
    for r, (tag, imgs) in enumerate(rows):
        y = pad_t + r * (c + gap)
        col = "#6b7280" if r == 0 else "#7c3aed"
        d.text((14, y + c // 2 - 9), tag, fill=col, font=_font(16, bold=True))
        for i, im in enumerate(imgs):
            canvas.paste(im.resize((c, c), Image.LANCZOS), (pad_l + i * c, y))
    canvas.save(args.out)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
