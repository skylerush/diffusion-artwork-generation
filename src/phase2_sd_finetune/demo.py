"""LIVE DEMO — base Stable Diffusion vs. OUR LoRA fine-tune, on the SAME prompt and SAME seed.

Crucially, the prompts here contain **no style word at all** (no "impressionist", no artist name).
Our evaluation prompts *did* contain "an impressionist painting of...", which let the base model
already look impressionist and masked what the fine-tuning actually added (see JOURNEY.md §7 —
this was flagged as the single best follow-up experiment).

So: any Impressionist character in the bottom row comes from the **fine-tuned weights**, not the prompt.

    python src/phase2_sd_finetune/demo.py
"""
import argparse
import pathlib
import sys

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import DEFAULT_BASE  # noqa: E402

PROMPTS = [
    "a river at sunset with tall poplar trees",
    "a harbour with sailing boats",
    "a woman with a parasol in a field of flowers",
    "a snowy village street in winter",
]


def _font(size, bold=False):
    for name in (("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate(pipe, prompts, seed, steps, guidance, device):
    out = []
    for i, p in enumerate(prompts):
        g = torch.Generator(device=device).manual_seed(seed + i)
        out.append(pipe(p, num_inference_steps=steps, guidance_scale=guidance, generator=g).images[0])
        print(f"    [{i+1}/{len(prompts)}] {p}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=str(ROOT / "outputs" / "phase2" / "lora_r16" / "ckpt" / "lora_last.pt"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--cell", type=int, default=384)
    ap.add_argument("--out", default=str(ROOT / "report" / "figures" / "demo_base_vs_lora.png"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading base Stable Diffusion 1.5 ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(DEFAULT_BASE, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print("\n[1/2] generating with BASE SD-1.5 (no fine-tune):", flush=True)
    base_imgs = generate(pipe, PROMPTS, args.seed, args.steps, args.guidance, device)

    print(f"\nattaching OUR LoRA adapter (rank {args.rank}) ...", flush=True)
    pipe.unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                     target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    set_peft_model_state_dict(pipe.unet, torch.load(args.lora, map_location="cpu"))

    print("\n[2/2] generating with OUR LoRA (same prompts, same seeds):", flush=True)
    lora_imgs = generate(pipe, PROMPTS, args.seed, args.steps, args.guidance, device)

    # ---- compose a labelled 2-row comparison grid ----
    c, pad_l, pad_t, gap = args.cell, 190, 76, 30
    W = pad_l + c * len(PROMPTS)
    H = pad_t + 2 * c + gap
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)

    d.text((14, 12), "Same prompt · same seed · NO style word in the prompt",
           fill="black", font=_font(20, bold=True))
    d.text((14, 40), "Any Impressionist character in the bottom row comes from the fine-tuned weights.",
           fill="#4b5563", font=_font(14))

    for i, p in enumerate(PROMPTS):
        d.text((pad_l + i * c + 8, pad_t - 20), f'"{p}"', fill="#374151", font=_font(12))

    rows = [("Base SD-1.5", "(no fine-tune)", base_imgs, "#6b7280"),
            ("+ our LoRA r16", "(Impressionism)", lora_imgs, "#7c3aed")]
    for r, (title, sub, imgs, col) in enumerate(rows):
        y = pad_t + r * (c + gap)
        d.text((14, y + c // 2 - 22), title, fill=col, font=_font(17, bold=True))
        d.text((14, y + c // 2 + 2), sub, fill=col, font=_font(13))
        for i, im in enumerate(imgs):
            canvas.paste(im.resize((c, c), Image.LANCZOS), (pad_l + i * c, y))

    canvas.save(args.out)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
