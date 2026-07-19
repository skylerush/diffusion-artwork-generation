"""A from-scratch U-Net noise predictor for DDPM.

Implements the standard components ourselves (no diffusers): sinusoidal timestep
embedding, residual blocks with FiLM-style time conditioning, self-attention at
chosen resolutions, and a symmetric down/up path with skip connections.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.out_layers = nn.Sequential(
            norm(out_ch), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.in_layers(x)
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.out_layers(h)
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        assert ch % num_heads == 0, f"channels {ch} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        self.norm = norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W).unbind(1)
        scale = 1.0 / math.sqrt(q.shape[-2])
        attn = torch.einsum("bhcn,bhcm->bhnm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=128, ch_mults=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), image_size=64, dropout=0.0, num_heads=4):
        super().__init__()
        self.base = base
        t_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # ---- down path ----
        self.downs = nn.ModuleList()
        chs = [base]
        cur = base
        res = image_size
        for i, mult in enumerate(ch_mults):
            out = base * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(cur, out, t_dim, dropout)]
                cur = out
                if res in attn_resolutions:
                    layers.append(AttnBlock(cur, num_heads))
                self.downs.append(nn.ModuleList(layers))
                chs.append(cur)
            if i != len(ch_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(cur)]))
                chs.append(cur)
                res //= 2

        # ---- middle ----
        self.mid = nn.ModuleList([
            ResBlock(cur, cur, t_dim, dropout),
            AttnBlock(cur, num_heads),
            ResBlock(cur, cur, t_dim, dropout),
        ])

        # ---- up path ----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out = base * mult
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(cur + chs.pop(), out, t_dim, dropout)]
                cur = out
                if res in attn_resolutions:
                    layers.append(AttnBlock(cur, num_heads))
                if i != 0 and j == num_res_blocks:
                    layers.append(Upsample(cur))
                    res *= 2
                self.ups.append(nn.ModuleList(layers))

        self.out = nn.Sequential(norm(cur), nn.SiLU(), nn.Conv2d(cur, in_ch, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.in_conv(x)
        hs = [h]
        for block in self.downs:
            if isinstance(block[0], Downsample):
                h = block[0](h)
            else:
                h = block[0](h, t_emb)
                for layer in block[1:]:
                    h = layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
        for block in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block[0](h, t_emb)
            for layer in block[1:]:
                h = layer(h)
        return self.out(h)
