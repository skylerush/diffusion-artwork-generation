"""Train vs held-out validation loss for every Phase-2 model — the diffusion analogue of a
train/validation accuracy comparison.

WHY THIS SCRIPT EXISTS
----------------------
There is no "validation AUC" or "validation accuracy" for a generative model: both metrics compare a
prediction against a known-correct answer, and there is no correct image for a given prompt. What
*is* well defined is the model's own training objective — the epsilon-prediction MSE — evaluated on
images it never saw. That is a real generalisation measure:

    train loss  = eps-MSE on images the model was fine-tuned on
    val loss    = eps-MSE on the 300 held-out paintings, never trained on
    gap         = val - train   ->  larger gap = more overfitting

This is the question our FID comparison could not settle: does a higher-rank LoRA overfit? Nothing
in the project logged validation loss during training, so we compute it post-hoc from checkpoints.

FAIRNESS
--------
Every model is scored on the SAME images with the SAME (timestep, noise) draws, fixed by seed. The
eps-MSE is stochastic in t and in the noise, so without this the differences between models would be
swamped by sampling noise. We average over --repeats draws per image.

The held-out split has no captions (prepare_data.py only writes metadata.jsonl for train), so BOTH
splits are scored under one fixed caption. The absolute value therefore is not comparable to the
training-time loss curves, but the train-vs-val GAP — the thing we care about — is valid, because
both sides get identical treatment.

    python src/phase2_sd_finetune/eval_heldout_loss.py
"""
import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer                     # noqa: E402
from peft import LoraConfig                                               # noqa: E402
from peft.utils import set_peft_model_state_dict                          # noqa: E402
from src.phase2_sd_finetune.train_lora import DEFAULT_BASE                # noqa: E402

P2 = ROOT / "outputs" / "phase2"
CAPTION = "an impressionist painting"

MODELS = [
    ("base",            {"kind": "base"}),
    ("lora_r4",         {"kind": "lora", "rank": 4,  "path": P2 / "lora_r4" / "ckpt" / "lora_last.pt"}),
    ("lora_r16",        {"kind": "lora", "rank": 16, "path": P2 / "lora_r16" / "ckpt" / "lora_last.pt"}),
    ("lora_r64",        {"kind": "lora", "rank": 64, "path": P2 / "lora_r64" / "ckpt" / "lora_last.pt"}),
    ("dreambooth",      {"kind": "lora", "rank": 16, "path": P2 / "dreambooth" / "ckpt" / "lora_last.pt"}),
    ("full_ft",         {"kind": "unet", "path": P2 / "full_ft" / "ckpt" / "unet_last"}),
    ("full_ft_matched", {"kind": "unet", "path": P2 / "full_ft_matched" / "ckpt" / "unet_last"}),
]
EXTS = {".jpg", ".jpeg", ".png"}


def load_split(folder, n, size=512):
    paths = sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in EXTS)[:n]
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    return paths, tf


@torch.no_grad()
def encode_latents(vae, paths, tf, device, bs=8):
    """VAE-encode a split ONCE; every model then reuses the identical latents."""
    out = []
    for i in range(0, len(paths), bs):
        px = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device)
        out.append((vae.encode(px).latent_dist.mean * vae.config.scaling_factor).cpu())
    return torch.cat(out)


@torch.no_grad()
def split_loss(unet, latents, enc, sched, device, repeats, seed, bs=8):
    """Mean eps-MSE over fixed (t, noise) draws. Identical draws for every model via `seed`."""
    total, count = 0.0, 0
    for r in range(repeats):
        g = torch.Generator(device="cpu").manual_seed(seed + r)
        ts_all = torch.randint(0, sched.config.num_train_timesteps, (latents.shape[0],), generator=g)
        noise_all = torch.randn(latents.shape, generator=g)
        for i in range(0, latents.shape[0], bs):
            lat = latents[i:i + bs].to(device)
            noise = noise_all[i:i + bs].to(device)
            ts = ts_all[i:i + bs].to(device)
            noisy = sched.add_noise(lat, noise, ts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(noisy, ts, encoder_hidden_states=enc[:lat.shape[0]]).sample
            total += F.mse_loss(pred.float(), noise.float()).item() * lat.shape[0]
            count += lat.shape[0]
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="images per split")
    ap.add_argument("--repeats", type=int, default=4, help="(t, noise) draws per image")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "heldout_loss.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading SD-1.5 components ...", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(DEFAULT_BASE, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(DEFAULT_BASE, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(DEFAULT_BASE, subfolder="vae").to(device)
    sched = DDPMScheduler.from_pretrained(DEFAULT_BASE, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    with torch.no_grad():
        ids = tokenizer(CAPTION, max_length=tokenizer.model_max_length, padding="max_length",
                        truncation=True, return_tensors="pt").input_ids.to(device)
        enc = text_encoder(ids)[0].repeat(args.batch, 1, 1)

    splits = {}
    for name, folder in (("train", ROOT / "data" / "impressionism_512" / "train"),
                         ("val",   ROOT / "data" / "impressionism_512" / "heldout")):
        paths, tf = load_split(folder, args.n)
        print(f"  {name}: {len(paths)} images -> encoding latents once", flush=True)
        splits[name] = encode_latents(vae, paths, tf, device, args.batch)
    del vae
    torch.cuda.empty_cache()

    results = {"caption": CAPTION, "n_per_split": args.n, "repeats": args.repeats,
               "seed": args.seed, "models": {}}
    for name, spec in MODELS:
        if spec["kind"] != "base" and not pathlib.Path(spec["path"]).exists():
            print(f"  SKIP {name} (checkpoint missing)", flush=True)
            continue
        print(f"\n[{name}] loading ...", flush=True)
        if spec["kind"] == "unet":
            unet = UNet2DConditionModel.from_pretrained(spec["path"]).to(device)
        else:
            unet = UNet2DConditionModel.from_pretrained(DEFAULT_BASE, subfolder="unet").to(device)
            if spec["kind"] == "lora":
                unet.add_adapter(LoraConfig(r=spec["rank"], lora_alpha=spec["rank"],
                                            init_lora_weights="gaussian",
                                            target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
                set_peft_model_state_dict(unet, torch.load(spec["path"], map_location="cpu"))
        unet.eval()

        tr = split_loss(unet, splits["train"], enc, sched, device, args.repeats, args.seed, args.batch)
        va = split_loss(unet, splits["val"], enc, sched, device, args.repeats, args.seed, args.batch)
        results["models"][name] = {"train": tr, "val": va, "gap": va - tr}
        print(f"[{name}]  train {tr:.5f}   val {va:.5f}   gap {va - tr:+.5f}", flush=True)

        del unet
        torch.cuda.empty_cache()

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")
    print(f"\n  {'model':<18}{'train':>10}{'val':>10}{'gap':>10}")
    print("  " + "-" * 48)
    for k, v in results["models"].items():
        print(f"  {k:<18}{v['train']:>10.5f}{v['val']:>10.5f}{v['gap']:>+10.5f}")
    print("\n  gap = val - train. Larger gap = the model fits its training set better than unseen")
    print("  paintings, i.e. more overfitting. This is the honest analogue of a train/val")
    print("  accuracy split — 'AUC' does not exist for a generative model.")


if __name__ == "__main__":
    main()
