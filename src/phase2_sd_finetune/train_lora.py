"""Phase-2: LoRA fine-tuning of Stable Diffusion v1.5 on Impressionism (WikiArt).

Adds low-rank adapters (peft) to the UNet attention projections; VAE, text encoder and
UNet base weights stay frozen. Standard latent-diffusion training: encode image -> latent,
add noise (SD scheduler), predict the noise with text conditioning. The --rank sweep here
is central to our Phase-2 comparison (LoRA vs full fine-tune vs DreamBooth).

Smoke test (pulls SD-1.5, 6 steps on the tiny _imp_test set):
    python src/phase2_sd_finetune/train_lora.py --smoke
Real run:
    python src/phase2_sd_finetune/train_lora.py --rank 16 --steps 1500 --run-name lora_r16
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
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diffusers import (AutoencoderKL, UNet2DConditionModel, DDPMScheduler,  # noqa: E402
                       StableDiffusionPipeline, DPMSolverMultistepScheduler)
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict  # noqa: E402

DEFAULT_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"

SAMPLE_PROMPTS = [
    "an impressionist painting of a landscape, in the style of Claude-Monet",
    "an impressionist painting of a harbor at sunset",
    "an impressionist painting of a woman in a garden",
    "an impressionist painting of a snowy village street",
]


class ImageCaptionDataset(Dataset):
    def __init__(self, root, size, tokenizer):
        self.root = pathlib.Path(root)
        with open(self.root / "metadata.jsonl", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        self.tok = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.root / r["file_name"]).convert("RGB")
        ids = self.tok(r["text"], max_length=self.tok.model_max_length,
                       padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        return {"pixel_values": self.tf(img), "input_ids": ids}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "impressionism_512" / "train"))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default = rank)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-util", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.data = str(ROOT / "data" / "_imp_test" / "train")
        args.rank, args.steps, args.batch_size, args.grad_accum = 4, 6, 1, 1
        args.sample_every, args.ckpt_every, args.log_every = 6, 6, 1
    if args.alpha is None:
        args.alpha = args.rank
    if args.run_name is None:
        args.run_name = f"lora_r{args.rank}"
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = ROOT / "outputs" / "phase2" / args.run_name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    print("loading SD-1.5 components from", args.base, flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet").to(device)
    noise_sched = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
                          target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(lora_cfg)
    unet.enable_gradient_checkpointing()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA params: {sum(p.numel() for p in lora_params)/1e6:.3f}M "
          f"(rank {args.rank}, alpha {args.alpha})", flush=True)
    opt = torch.optim.AdamW(lora_params, lr=args.lr)

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
    unet.train()
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
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
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
            torch.save(get_peft_model_state_dict(unet), out / "ckpt" / "lora_last.pt")

    print("LORA_TRAIN_DONE", flush=True)


@torch.no_grad()
def run_sample(args, out, step, vae, text_encoder, tokenizer, unet, device):
    unet.eval()
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(args.base, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator(device=device).manual_seed(args.seed)
    images = pipe(SAMPLE_PROMPTS, num_inference_steps=30, guidance_scale=7.0, generator=g).images
    tens = torch.stack([TF.to_tensor(im) for im in images])
    save_image(tens, str(out / "samples" / f"sample_{step:06d}.png"), nrow=2)
    print(f"  saved samples @ step {step}", flush=True)


if __name__ == "__main__":
    main()
