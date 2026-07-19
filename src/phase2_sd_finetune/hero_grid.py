"""HERO GRID — the same prompt through EVERY model we trained (8 columns x 2 prompts).

The presentation visual: one prompt, same seed, eight models side by side.
    python src/phase2_sd_finetune/hero_grid.py  ->  report/figures/hero_grid_8models.png
"""
import pathlib
import sys

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import DEFAULT_BASE   # noqa: E402
from src.phase2_sd_finetune.infer import set_lora_scale      # noqa: E402
from src.phase2_sd_finetune.demo import _font                # noqa: E402

P2 = ROOT / "outputs" / "phase2"
# (label, kind, path, rank, scale)
MODELS = [
    ("Base SD-1.5\n(no fine-tune)",      "base", None, None, None),
    ("LoRA r4 @1.5\n3.3 MB",             "lora", P2 / "lora_r4" / "ckpt" / "lora_last.pt", 4, 1.5),
    ("LoRA r16 @1.0\n12.8 MB",           "lora", P2 / "lora_r16" / "ckpt" / "lora_last.pt", 16, 1.0),
    ("LoRA r16 @1.5\n12.8 MB  (BEST)",   "lora", P2 / "lora_r16" / "ckpt" / "lora_last.pt", 16, 1.5),
    ("LoRA r64 @1.5\n51 MB",             "lora", P2 / "lora_r64" / "ckpt" / "lora_last.pt", 64, 1.5),
    ("DreamBooth @1.5\n12.8 MB (sks)",   "lora", P2 / "dreambooth" / "ckpt" / "lora_last.pt", 16, 1.5),
    ("Full FT\n3.44 GB",                 "unet", P2 / "full_ft" / "ckpt" / "unet_last", None, None),
    ("Full FT matched\n3.44 GB",         "unet", P2 / "full_ft_matched" / "ckpt" / "unet_last", None, None),
]
PROMPTS = [
    "a harbour with sailing boats at sunset",
    "a woman with a parasol in a field of flowers",
]
SEED, STEPS, GUIDE, CELL = 1234, 25, 7.0, 288


def fresh_unet(device):
    return UNet2DConditionModel.from_pretrained(DEFAULT_BASE, subfolder="unet").to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading pipeline ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(DEFAULT_BASE, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    results = {}  # (model_idx, prompt_idx) -> PIL
    for mi, (label, kind, path, rank, scale) in enumerate(MODELS):
        print(f"[{mi+1}/{len(MODELS)}] {label.splitlines()[0]}", flush=True)
        old = pipe.unet
        if kind == "base":
            pipe.unet = fresh_unet(device)
        elif kind == "lora":
            u = fresh_unet(device)
            u.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                     target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
            set_peft_model_state_dict(u, torch.load(path, map_location="cpu"))
            if scale and scale != 1.0:
                set_lora_scale(u, scale)
            pipe.unet = u
        else:
            pipe.unet = UNet2DConditionModel.from_pretrained(path).to(device)
        del old
        torch.cuda.empty_cache()

        for pi, prompt in enumerate(PROMPTS):
            g = torch.Generator(device=device).manual_seed(SEED + pi)
            results[(mi, pi)] = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDE,
                                     generator=g).images[0]
            print(f"    prompt {pi+1} done", flush=True)

    # ---- compose ----
    pad_l, pad_t = 200, 64
    W = pad_l + CELL * len(MODELS)
    H = pad_t + CELL * len(PROMPTS)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), "One prompt, every model — same seed per row", fill="black", font=_font(20, bold=True))
    for mi, (label, *_ ) in enumerate(MODELS):
        d.multiline_text((pad_l + mi * CELL + 6, pad_t - 40), label, fill="#374151",
                         font=_font(12, bold=True), spacing=2)
    for pi, prompt in enumerate(PROMPTS):
        y = pad_t + pi * CELL
        d.multiline_text((10, y + CELL // 2 - 20), f'"{prompt}"'.replace(" with ", "\nwith "),
                         fill="#111827", font=_font(12), spacing=2)
        for mi in range(len(MODELS)):
            canvas.paste(results[(mi, pi)].resize((CELL, CELL), Image.LANCZOS),
                         (pad_l + mi * CELL, y))
    out = ROOT / "report" / "figures" / "hero_grid_8models.png"
    canvas.save(out)
    print(f"HERO_GRID_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
