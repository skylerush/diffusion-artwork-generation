"""Exponential Moving Average of model weights, with warmup.

A shadow copy of the model is nudged toward the live weights after every step. Sampling
from the EMA weights is standard in diffusion models and stabilises sample quality.

Warmup matters: a fixed high decay (e.g. 0.9999) averages over ~1/(1-decay)=10k steps, so
for short runs the shadow stays dominated by the *random initialisation* and samples look
like noise. We ramp the decay as (1+t)/(10+t) → max_decay, tracking fast early, smoothing late.
"""
import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmup: bool = True):
        self.max_decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _decay(self) -> float:
        if not self.warmup:
            return self.max_decay
        return min(self.max_decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = self._decay()
        # Fused foreach update: 2 kernels instead of 2*N (matters at every training step).
        s_params = list(self.shadow.parameters())
        m_params = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(s_params, d)
        torch._foreach_add_(s_params, m_params, alpha=1.0 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
