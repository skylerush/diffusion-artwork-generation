"""Phase-2: FULL fine-tuning of the Stable Diffusion v1.5 UNet on Impressionism.

All UNet parameters are trained at a LOW lr (to avoid catastrophic forgetting); VAE +
text encoder stay frozen. Gradient checkpointing keeps all 860M params trainable within
32 GB at 512px. This is the 'best achievable quality' baseline to contrast with cheap LoRA.
Reuses the validated dataset + sampling from train_lora.py.

Smoke:  python src/phase2_sd_finetune/train_full.py --smoke
Real:   python src/phase2_sd_finetune/train_full.py --steps 1500 --lr 1e-6 --run-name full_ft
Fail-ladder variant: --lr 1e-4  (expect catastrophic forgetting)
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.phase2_sd_finetune.train_lora import ImageCaptionDataset, run_sample, DEFAULT_BASE  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "impressionism_512" / "train"))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-util", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--run-name", default="full_ft")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.data = str(ROOT / "data" / "_imp_test" / "train")
        args.steps, args.batch_size, args.grad_accum = 6, 1, 1
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
    unet.requires_grad_(True)
    unet.enable_gradient_checkpointing()
    unet.train()

    params = list(unet.parameters())
    print(f"trainable UNet params: {sum(p.numel() for p in params)/1e6:.1f}M (FULL fine-tune)", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr)

    ds = ImageCaptionDataset(args.data, args.size, tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"dataset: {len(ds)} image-caption pairs", flush=True)

    def cycle(loader):
        while True:
            for b in loader:
                yield b
    it = cycle(dl)

    throttle = (device == "cuda") and (0.0 < args.max_util < 1.0)
    scaling = vae.config.scaling_factor
    t0 = time.time()
    for step in range(args.steps):
        if throttle:
            torch.cuda.synchronize()
            tic = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            batch = next(it)
            px = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                latents = vae.encode(px).latent_dist.sample() * scaling
                enc = text_encoder(ids)[0]
            noise = torch.randn_like(latents)
            ts = torch.randint(0, noise_sched.config.num_train_timesteps,
                               (latents.shape[0],), device=device).long()
            noisy = noise_sched.add_noise(latents, noise, ts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(noisy, ts, encoder_hidden_states=enc).sample
            loss = F.mse_loss(pred.float(), noise.float()) / args.grad_accum
            loss.backward()
            accum += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if throttle:
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - tic) * (1.0 / args.max_util - 1.0))

        if step % args.log_every == 0:
            ips = (step + 1) * args.batch_size * args.grad_accum / (time.time() - t0)
            print(f"step {step:>5} | loss {accum:.4f} | {ips:.2f} img/s", flush=True)

        last = step == args.steps - 1
        if (step + 1) % args.sample_every == 0 or last:
            run_sample(args, out, step + 1, vae, text_encoder, tokenizer, unet, device)
            unet.train()
        if (step + 1) % args.ckpt_every == 0 or last:
            unet.save_pretrained(out / "ckpt" / "unet_last")

    print("FULL_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
