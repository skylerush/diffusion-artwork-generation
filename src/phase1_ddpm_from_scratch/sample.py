"""Generate a grid of samples from a trained Phase-1 checkpoint (EMA weights)."""
import argparse
import pathlib
import sys

import torch
import torchvision

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.phase1_ddpm_from_scratch.unet import UNet                    # noqa: E402
from src.phase1_ddpm_from_scratch.diffusion import GaussianDiffusion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    a = ck["args"]
    model = UNet(base=a["base"], ch_mults=tuple(a["ch_mults"]), num_res_blocks=a["num_res_blocks"],
                 attn_resolutions=tuple(a["attn_res"]), image_size=a["image_size"]).to(device)
    sd = ck["model"] if args.no_ema else ck.get("ema", ck["model"])
    model.load_state_dict(sd)
    model.eval()

    diffusion = GaussianDiffusion(timesteps=a["timesteps"], schedule=a["schedule"]).to(device)
    shp = (args.n, 3, a["image_size"], a["image_size"])
    with torch.no_grad():
        if args.sampler == "ddim":
            x = diffusion.ddim_sample(model, shp, device, steps=args.ddim_steps)
        else:
            x = diffusion.sample(model, shp, device)

    out = args.out or str(pathlib.Path(args.ckpt).parents[1] / f"sample_{args.sampler}.png")
    torchvision.utils.save_image((x.clamp(-1, 1) + 1) / 2, out, nrow=int(args.n ** 0.5))
    print("saved", out)


if __name__ == "__main__":
    main()
