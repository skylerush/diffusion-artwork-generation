"""Phase-2: DreamBooth (LoRA) — style personalization with prior preservation.

Instance images (our Impressionism set) are tied to a rare-token instance prompt; with
`--with-prior`, we also generate class images from the BASE model for a generic class
prompt and add a prior-preservation loss so the model keeps its generality instead of
overfitting / drifting. Failure-ladder point: WITHOUT prior preservation it overfits;
WITH it, generalisation improves.

Smoke:  python src/phase2_sd_finetune/train_dreambooth.py --smoke
Real:   python src/phase2_sd_finetune/train_dreambooth.py --with-prior --steps 1200 --run-name dreambooth
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from diffusers import (AutoencoderKL, UNet2DConditionModel, DDPMScheduler,
                       StableDiffusionPipeline, DPMSolverMultistepScheduler)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import run_sample, DEFAULT_BASE  # noqa: E402

INSTANCE_PROMPT = "a painting in sks impressionist style"
CLASS_PROMPT = "a painting"
EXTS = {".jpg", ".jpeg", ".png"}


class PromptImageDataset(Dataset):
    """All images in `folder` paired with one fixed `prompt` (DreamBooth style)."""

    def __init__(self, folder, prompt, size, tokenizer):
        self.paths = sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in EXTS)
        self.tf = transforms.Compose([
            transforms.Resize(size), transforms.CenterCrop(size), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])
        self.ids = tokenizer(prompt, max_length=tokenizer.model_max_length,
                             padding="max_length", truncation=True, return_tensors="pt").input_ids[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"pixel_values": self.tf(Image.open(self.paths[i]).convert("RGB")), "input_ids": self.ids}


def gen_class_images(base, prompt, n, out_dir, device):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.glob("*.jpg")]
    if len(existing) >= n:
        return
    pipe = StableDiffusionPipeline.from_pretrained(base, safety_checker=None, requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    for i in range(len(existing), n):
        img = pipe(prompt, num_inference_steps=25, guidance_scale=7.0).images[0]
        img.save(out_dir / f"class_{i:04d}.jpg", quality=95)
    del pipe
    torch.cuda.empty_cache()
    print(f"generated {n} class images for prior preservation", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "impressionism_512" / "train"))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-util", type=float, default=1.0)
    ap.add_argument("--with-prior", action="store_true", help="enable prior-preservation loss")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--num-class-images", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--run-name", default="dreambooth")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.data = str(ROOT / "data" / "_imp_test" / "train")
        args.rank, args.steps, args.batch_size, args.grad_accum = 4, 6, 1, 1
        args.num_class_images = 2
        args.sample_every, args.ckpt_every, args.log_every = 6, 6, 1
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = ROOT / "outputs" / "phase2" / args.run_name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    print("loading SD-1.5 from", args.base, flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet").to(device)
    noise_sched = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    inst = PromptImageDataset(args.data, INSTANCE_PROMPT, args.size, tokenizer)
    inst_dl = DataLoader(inst, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"instance images: {len(inst)}  | prior preservation: {args.with_prior}", flush=True)

    class_it = None
    if args.with_prior:
        class_dir = ROOT / "data" / "_dreambooth_class"
        gen_class_images(args.base, CLASS_PROMPT, args.num_class_images, class_dir, device)
        cls = PromptImageDataset(class_dir, CLASS_PROMPT, args.size, tokenizer)
        class_dl = DataLoader(cls, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        def cyc(loader):
            while True:
                for b in loader:
                    yield b
        class_it = cyc(class_dl)

    def cyc(loader):
        while True:
            for b in loader:
                yield b
    inst_it = cyc(inst_dl)

    throttle = (device == "cuda") and (0.0 < args.max_util < 1.0)
    scaling = vae.config.scaling_factor

    def loss_for(batch):
        px = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        with torch.no_grad():
            latents = vae.encode(px).latent_dist.sample() * scaling
            enc = text_encoder(ids)[0]
        noise = torch.randn_like(latents)
        ts = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy = noise_sched.add_noise(latents, noise, ts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(noisy, ts, encoder_hidden_states=enc).sample
        return F.mse_loss(pred.float(), noise.float())

    unet.train()
    t0 = time.time()
    for step in range(args.steps):
        if throttle:
            torch.cuda.synchronize()
            tic = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            loss = loss_for(next(inst_it))
            if class_it is not None:
                loss = loss + args.prior_weight * loss_for(next(class_it))
            loss = loss / args.grad_accum
            loss.backward()
            accum += loss.item()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if throttle:
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - tic) * (1.0 / args.max_util - 1.0))

        if step % args.log_every == 0:
            print(f"step {step:>5} | loss {accum:.4f} | {(step+1)/(time.time()-t0):.2f} it/s", flush=True)
        last = step == args.steps - 1
        if (step + 1) % args.sample_every == 0 or last:
            run_sample(args, out, step + 1, vae, text_encoder, tokenizer, unet, device)
            unet.train()
        if (step + 1) % args.ckpt_every == 0 or last:
            torch.save(get_peft_model_state_dict(unet), out / "ckpt" / "lora_last.pt")

    print("DREAMBOOTH_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
