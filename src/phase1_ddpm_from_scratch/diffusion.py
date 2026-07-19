"""Gaussian diffusion: forward noising, training loss, and DDPM/DDIM sampling.

Follows Ho et al. (2020) for the forward process and ancestral sampler, and
Song et al. (2020) for the deterministic DDIM sampler. ε-prediction objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.schedules import make_beta_schedule


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Gather per-sample schedule values a[t] and reshape to broadcast over x."""
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", schedule_kwargs=None):
        super().__init__()
        self.timesteps = timesteps
        betas = make_beta_schedule(schedule, timesteps, **(schedule_kwargs or {})).float()
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        reg = self.register_buffer
        reg("betas", betas)
        reg("alphas", alphas)
        reg("acp", acp)
        reg("acp_prev", acp_prev)
        reg("sqrt_acp", torch.sqrt(acp))
        reg("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))
        reg("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        reg("sqrt_recip_acp", torch.sqrt(1.0 / acp))
        reg("sqrt_recipm1_acp", torch.sqrt(1.0 / acp - 1.0))
        reg("posterior_variance", betas * (1.0 - acp_prev) / (1.0 - acp))

    # ---- forward process ----
    def q_sample(self, x0, t, noise):
        return (extract(self.sqrt_acp, t, x0.shape) * x0
                + extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def p_losses(self, model, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t)
        return F.mse_loss(pred.float(), noise.float())

    def predict_x0(self, x_t, t, eps):
        return (extract(self.sqrt_recip_acp, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_acp, t, x_t.shape) * eps)

    # ---- reverse process: ancestral DDPM ----
    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        eps = model(x_t, t)
        coef = extract(self.betas, t, x_t.shape) / extract(self.sqrt_one_minus_acp, t, x_t.shape)
        mean = extract(self.sqrt_recip_alphas, t, x_t.shape) * (x_t - coef * eps)
        if int(t[0]) == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(extract(self.posterior_variance, t, x_t.shape)) * noise

    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    # ---- reverse process: deterministic DDIM ----
    @torch.no_grad()
    def ddim_sample(self, model, shape, device, steps=50, eta=0.0):
        times = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        x = torch.randn(shape, device=device)
        for idx, ti in enumerate(times):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            eps = model(x, t)
            x0 = self.predict_x0(x, t, eps).clamp(-1, 1)
            if idx == len(times) - 1:
                x = x0
                continue
            t_next = torch.full((shape[0],), times[idx + 1], device=device, dtype=torch.long)
            acp_t = extract(self.acp, t, x.shape)
            acp_next = extract(self.acp, t_next, x.shape)
            sigma = eta * torch.sqrt((1 - acp_next) / (1 - acp_t) * (1 - acp_t / acp_next))
            dir_xt = torch.sqrt((1 - acp_next - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(acp_next) * x0 + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x
