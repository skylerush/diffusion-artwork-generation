"""PLAYGROUND — type a prompt, get an image. The simplest way to use our fine-tuned models.

Examples (from the project root):
    python src/phase2_sd_finetune/generate.py "a harbour with sailing boats at sunset"
    python src/phase2_sd_finetune/generate.py "a rainy boulevard" --model base          # no fine-tune
    python src/phase2_sd_finetune/generate.py "a garden in spring" --n 4 --scale 2.0
    python src/phase2_sd_finetune/generate.py "a painting in sks impressionist style of a lake" --model dreambooth

Defaults: our best model (LoRA r16) at the tuned strength (scale 1.5), 1 image, 512px.
Output lands in outputs/playground/.
"""
import argparse
import math
import pathlib
import re
import sys

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import DEFAULT_BASE      # noqa: E402
from src.phase2_sd_finetune.infer import set_lora_scale         # noqa: E402

P2 = ROOT / "outputs" / "phase2"
MODELS = {
    "base":            {"kind": "base"},
    "lora_r4":         {"kind": "lora", "rank": 4,  "path": P2 / "lora_r4" / "ckpt" / "lora_last.pt"},
    "lora_r16":        {"kind": "lora", "rank": 16, "path": P2 / "lora_r16" / "ckpt" / "lora_last.pt"},
    "lora_r64":        {"kind": "lora", "rank": 64, "path": P2 / "lora_r64" / "ckpt" / "lora_last.pt"},
    "dreambooth":      {"kind": "lora", "rank": 16, "path": P2 / "dreambooth" / "ckpt" / "lora_last.pt"},
    "full_ft":         {"kind": "unet", "path": P2 / "full_ft" / "ckpt" / "unet_last"},
    "full_ft_matched": {"kind": "unet", "path": P2 / "full_ft_matched" / "ckpt" / "unet_last"},
}


def main():
    ap = argparse.ArgumentParser(description="Prompt our fine-tuned Impressionism models.")
    ap.add_argument("prompt", nargs="+", help="the text prompt (quotes optional)")
    ap.add_argument("--model", default="lora_r16", choices=sorted(MODELS),
                    help="which trained model to use (default: lora_r16, our best)")
    ap.add_argument("--scale", type=float, default=1.5,
                    help="LoRA strength; 1.0 subtle, 1.5 tuned default, 2.0 strong (LoRA models only)")
    ap.add_argument("--n", type=int, default=1, help="how many images (saved as one grid if >1)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prompt = " ".join(args.prompt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = MODELS[args.model]

    print(f"loading Stable Diffusion 1.5 + model '{args.model}' ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(DEFAULT_BASE, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if spec["kind"] == "lora":
        if not spec["path"].exists():
            raise SystemExit(f"checkpoint missing: {spec['path']}")
        pipe.unet.add_adapter(LoraConfig(r=spec["rank"], lora_alpha=spec["rank"],
                                         init_lora_weights="gaussian",
                                         target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        set_peft_model_state_dict(pipe.unet, torch.load(spec["path"], map_location="cpu"))
        if args.scale != 1.0:
            set_lora_scale(pipe.unet, args.scale)
        print(f"  -> LoRA rank {spec['rank']} @ strength {args.scale}", flush=True)
    elif spec["kind"] == "unet":
        if not spec["path"].exists():
            raise SystemExit(f"checkpoint missing: {spec['path']}")
        pipe.unet = UNet2DConditionModel.from_pretrained(spec["path"])
        print("  -> full fine-tuned UNet (860M params)", flush=True)
    else:
        print("  -> BASE model (no fine-tune) for comparison", flush=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    images = []
    for i in range(args.n):
        g = torch.Generator(device=device).manual_seed(args.seed + i)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=args.size, width=args.size, generator=g).images[0]
        images.append(img)
        print(f"  generated {i+1}/{args.n}  (seed {args.seed + i})", flush=True)

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt).strip("_")[:48]
    out = pathlib.Path(args.out) if args.out else \
        ROOT / "outputs" / "playground" / f"{args.model}_{slug}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = torch.stack([TF.to_tensor(im) for im in images])
    save_image(grid, str(out), nrow=max(1, math.ceil(math.sqrt(args.n))))
    print(f"\nSAVED -> {out}", flush=True)


if __name__ == "__main__":
    main()
