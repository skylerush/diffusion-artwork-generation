"""Train the from-scratch DDPM.

Examples
--------
Smoke test (random data, tiny model, ~30 steps — validates the whole pipeline):
    python src/phase1_ddpm_from_scratch/train.py --smoke

Real run on the butterflies sanity set (Phase 1a):
    python src/phase1_ddpm_from_scratch/train.py --data butterflies --run-name p1a_butterflies

Impressionism-64 (Phase 1b), pointing at a prepared image folder:
    python src/phase1_ddpm_from_scratch/train.py --data data/impressionism_64 --run-name p1b_imp
"""
import argparse
import json
import pathlib
import sys
import time

import torch
import torchvision

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.seeding import set_seed          # noqa: E402
from src.common.ema import EMA                    # noqa: E402
from src.phase1_ddpm_from_scratch.unet import UNet            # noqa: E402
from src.phase1_ddpm_from_scratch.diffusion import GaussianDiffusion  # noqa: E402
from src.phase1_ddpm_from_scratch.data import load_cached_uint8  # noqa: E402


class GPUData:
    """Whole small dataset kept on the GPU as uint8; per-step batch = index + normalize + flip, on-device."""

    def __init__(self, uint8_tensor, device, hflip=True):
        self.data = uint8_tensor.to(device)
        self.device, self.hflip, self.n = device, hflip, uint8_tensor.shape[0]

    def sample(self, bs):
        idx = torch.randint(0, self.n, (bs,), device=self.device)
        x = self.data[idx].float().div_(127.5).sub_(1.0)
        if self.hflip:
            flip = torch.rand(bs, device=self.device) < 0.5
            x[flip] = torch.flip(x[flip], dims=[3])
        return x


def save_grid(x, path, nrow=4):
    torchvision.utils.save_image((x.clamp(-1, 1) + 1) / 2, str(path), nrow=nrow)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="butterflies")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--ch-mults", type=int, nargs="+", default=[1, 2, 2, 2])
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--attn-res", type=int, nargs="+", default=[16])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--max-util", type=float, default=1.0,
                    help="target GPU duty cycle in (0,1]; <1 throttles by sleeping between steps "
                         "(e.g. 0.65 frees ~35%% of the GPU)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-ema", action="store_true",
                    help="on resume, re-init EMA from the loaded model (fixes a contaminated EMA)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end pipeline test")
    args = ap.parse_args()
    if args.smoke:
        args.data = "fake"
        args.image_size, args.batch_size, args.base = 16, 8, 32
        args.ch_mults, args.num_res_blocks, args.attn_res = [1, 2], 1, [8]
        args.steps, args.sample_every, args.ckpt_every, args.log_every = 30, 15, 15, 5
        args.ema_decay, args.num_workers = 0.99, 0
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    out = pathlib.Path(args.out) if args.out else ROOT / "outputs" / "phase1" / args.run_name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out / "tb"))

    model = UNet(base=args.base, ch_mults=tuple(args.ch_mults), num_res_blocks=args.num_res_blocks,
                 attn_resolutions=tuple(args.attn_res), image_size=args.image_size,
                 dropout=args.dropout).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"device {device} | out {out}")

    if args.data == "fake":
        cache = (torch.rand(256, 3, args.image_size, args.image_size) * 255).to(torch.uint8)
    else:
        cache = load_cached_uint8(args.data, args.image_size, max_images=args.max_images)
    data = GPUData(cache, device)
    print(f"dataset: {data.n} images cached on {device}")

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        if args.reset_ema:
            ema = EMA(model, decay=args.ema_decay)  # re-seed shadow from the loaded (good) model
            print(f"resumed @ step {start_step}; EMA reset from current model")
        else:
            ema.load_state_dict(ck["ema"])
            print(f"resumed from {args.resume} @ step {start_step}")

    throttle = (device == "cuda") and (0.0 < args.max_util < 1.0)
    if throttle:
        print(f"GPU throttle ON: targeting ~{args.max_util*100:.0f}% duty cycle "
              f"(idling ~{(1-args.max_util)*100:.0f}% of wall-time between steps)")
    model.train()
    t0 = time.time()
    for step in range(start_step, args.steps):
        if throttle:
            torch.cuda.synchronize()
            _tic = time.perf_counter()
        x = data.sample(args.batch_size)
        t = torch.randint(0, args.timesteps, (x.shape[0],), device=device).long()

        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = diffusion.p_losses(model, x, t)
        else:
            loss = diffusion.p_losses(model, x, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if throttle:
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - _tic) * (1.0 / args.max_util - 1.0))

        if step % args.log_every == 0:
            ips = (step - start_step + 1) * x.shape[0] / (time.time() - t0)
            writer.add_scalar("loss", loss.item(), step)
            print(f"step {step:>7} | loss {loss.item():.4f} | {ips:6.1f} img/s")

        last = step == args.steps - 1
        if (step + 1) % args.sample_every == 0 or last:
            model.eval()
            with torch.no_grad():
                shp = (min(16, args.batch_size), 3, args.image_size, args.image_size)
                ema_grid = diffusion.ddim_sample(ema.shadow, shp, device, steps=args.ddim_steps)
                raw_grid = diffusion.ddim_sample(model, shp, device, steps=args.ddim_steps)
            save_grid(ema_grid, out / "samples" / f"ddim_ema_{step+1:07d}.png", nrow=4)
            save_grid(raw_grid, out / "samples" / f"ddim_raw_{step+1:07d}.png", nrow=4)
            model.train()
        if (step + 1) % args.ckpt_every == 0 or last:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "args": vars(args)}, out / "ckpt" / "last.pt")

    writer.close()
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
