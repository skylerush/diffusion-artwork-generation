"""Datasets for Phase 1.

- 'butterflies': HuggingFace `huggan/smithsonian_butterflies_subset` (~1k imgs),
  the standard easy target used to validate a from-scratch DDPM.
- a folder path: any flat/recursive directory of images (e.g. our Impressionism-64
  crop, reused here at 64px).

Small Phase-1 datasets are decoded + resized ONCE into an in-RAM uint8 tensor cache,
so the GPU is never starved by per-step PIL decoding (only a cheap flip+normalize
runs per item). Image column is auto-detected (we never assume the schema).
"""
import pathlib

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from PIL import Image


class CachedImageDataset(Dataset):
    """Decode/resize/center-crop once into a uint8 (N,3,H,W) cache; flip+normalize per item."""

    def __init__(self, pil_images, image_size, hflip=True):
        self.hflip = hflip
        tensors = []
        for im in pil_images:
            im = im.convert("RGB")
            im = TF.resize(im, image_size)
            im = TF.center_crop(im, [image_size, image_size])
            tensors.append(TF.pil_to_tensor(im))
        if not tensors:
            raise ValueError("no images to cache")
        self.data = torch.stack(tensors)  # uint8

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = self.data[i].float().div_(127.5).sub_(1.0)  # -> [-1, 1]
        if self.hflip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x


def _detect_image_col(ds):
    from datasets import Image as HFImage
    for name, feat in ds.features.items():
        if isinstance(feat, HFImage):
            return name
    for c in ("image", "img", "images"):
        if c in ds.column_names:
            return c
    raise ValueError(f"Could not find an image column in {ds.column_names}")


def _folder_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in pathlib.Path(root).rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return [Image.open(p) for p in paths]


def get_dataloader(source, image_size, batch_size, num_workers=0, shuffle=True, max_images=None):
    if source == "butterflies":
        from datasets import load_dataset
        ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        col = _detect_image_col(ds)
        imgs = [ds[i][col] for i in range(len(ds))]
    else:
        if not pathlib.Path(source).exists():
            raise FileNotFoundError(f"Image source {source!r} not found. Use 'butterflies' or a folder path.")
        imgs = _folder_images(source)
    if max_images:
        imgs = imgs[:max_images]
    dataset = CachedImageDataset(imgs, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True, pin_memory=True)


def load_cached_uint8(source, image_size, max_images=None):
    """Decode a small dataset once and return its (N,3,H,W) uint8 tensor.

    Used by the Phase-1 trainer to keep the whole set GPU-resident (fast sampling).
    """
    if source == "butterflies":
        from datasets import load_dataset
        ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        col = _detect_image_col(ds)
        imgs = [ds[i][col] for i in range(len(ds))]
    else:
        if not pathlib.Path(source).exists():
            raise FileNotFoundError(f"Image source {source!r} not found. Use 'butterflies' or a folder path.")
        imgs = _folder_images(source)
    if max_images:
        imgs = imgs[:max_images]
    return CachedImageDataset(imgs, image_size).data
