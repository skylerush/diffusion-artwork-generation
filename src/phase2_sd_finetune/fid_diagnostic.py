"""Is our FID trustworthy? A diagnostic — and the answer matters a lot.

FID fits a Gaussian to **2048-dimensional** InceptionV3 features and compares means + covariances.
Estimating a 2048×2048 covariance from N samples requires N >> 2048. We used **N = 256** → the sample
covariance is *rank-deficient* (rank ≤ N−1 = 255 out of 2048) and FID is heavily biased **upward**.

Two checks:

  [1] REAL vs REAL — split the held-out set in half and compute FID between two sets of *genuine*
      Impressionist paintings. That is the metric's **noise floor**: the FID a *perfect* generator
      could not beat at this sample size. If the floor is already huge, our model scores were
      measuring the estimator's bias, not the models.

  [2] FID vs N — recompute our model's FID at N = 64/128/256. If it moves a lot, the number is
      dominated by sample size, not quality.

    python src/phase2_sd_finetune/fid_diagnostic.py
"""
import argparse
import pathlib

import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png"}


def paths_in(folder):
    return sorted(p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def load(paths, size=299):
    return torch.stack([TF.pil_to_tensor(Image.open(p).convert("RGB").resize((size, size))) for p in paths])


def fid_between(real, fake, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    m = FrechetInceptionDistance(normalize=True).to(device)
    for imgs, is_real in ((real, True), (fake, False)):
        for i in range(0, imgs.shape[0], 32):
            m.update((imgs[i:i + 32].float() / 255.0).to(device), real=is_real)
    return float(m.compute())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=str(ROOT / "outputs" / "phase2" / "lora_r16" / "eval_samples"))
    ap.add_argument("--ref-dir", default=str(ROOT / "data" / "impressionism_512" / "heldout"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_paths, gen_paths = paths_in(args.ref_dir), paths_in(args.gen_dir)
    print(f"real held-out paintings: {len(ref_paths)}   |   our generated images: {len(gen_paths)}")
    print("FID fits a 2048-dim Gaussian; a reliable covariance needs N >> 2048.\n")

    # ---- [1] the metric's noise floor: REAL vs REAL ----
    half = len(ref_paths) // 2
    a, b = load(ref_paths[:half]), load(ref_paths[half:2 * half])
    floor = fid_between(a, b, device)
    print(f"[1] REAL vs REAL  ({half} genuine paintings vs {half} OTHER genuine paintings)")
    print(f"    FID = {floor:.1f}")
    print("    ^ This is the FLOOR. A PERFECT generator could not score better than this at this N.\n")

    # ---- [2] FID vs N ----
    refs = load(ref_paths)
    print("[2] our LoRA r16 vs all held-out paintings, as N grows:")
    for n in (64, 128, 256):
        if n > len(gen_paths):
            continue
        val = fid_between(refs, load(gen_paths[:n]), device)
        print(f"    N = {n:4d}   FID = {val:.1f}")

    print("\nInterpretation: if the REAL-vs-REAL floor is close to our model scores, then our reported")
    print("FIDs were dominated by small-sample bias — the metric could not distinguish the methods.")
    print("Fix: evaluate on >= 2048 images (ideally 5-10k), not 256.")


if __name__ == "__main__":
    main()
