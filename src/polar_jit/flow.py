from __future__ import annotations

import torch
from torch import nn


class ConditionalFlowMatcher(nn.Module):
    """Straight conditional flow matching following the JiT clean-prediction parameterization."""

    def __init__(self, model, p_mean=-0.8, p_std=0.8, noise_scale=1.0, t_eps=0.05):
        super().__init__()
        self.model, self.p_mean, self.p_std = model, p_mean, p_std
        self.noise_scale, self.t_eps = noise_scale, t_eps

    def sample_t(self, batch, device):
        return torch.sigmoid(torch.randn(batch, device=device) * self.p_std + self.p_mean)

    def forward(self, target, s0):
        t = self.sample_t(target.shape[0], target.device)
        shape = (target.shape[0],) + (1,) * (target.ndim - 1)
        t_view = t.view(shape)
        noise = torch.randn_like(target) * self.noise_scale
        x_t = t_view * target + (1 - t_view) * noise
        prediction = self.model(x_t, t, s0)
        velocity_target = target - noise
        velocity_prediction = (prediction["clean"] - x_t) / (1 - t_view).clamp_min(self.t_eps)
        return prediction, velocity_prediction, velocity_target, t

    @torch.no_grad()
    def sample(self, s0, steps=20, method="heun", seed=42):
        generator = torch.Generator(device=s0.device).manual_seed(seed)
        x = torch.randn((s0.shape[0], self.model.out_channels, *s0.shape[-2:]),
                        generator=generator, device=s0.device, dtype=s0.dtype) * self.noise_scale
        times = torch.linspace(0, 1, steps + 1, device=s0.device, dtype=s0.dtype)

        def velocity(state, scalar_t):
            t = torch.full((state.shape[0],), float(scalar_t), device=state.device, dtype=state.dtype)
            clean = self.model(state, t, s0)["clean"]
            return (clean - state) / (1 - scalar_t).clamp_min(self.t_eps)

        for index in range(steps):
            t0, t1 = times[index], times[index + 1]
            v0 = velocity(x, t0)
            euler = x + (t1 - t0) * v0
            if method == "euler" or index == steps - 1:
                x = euler
            elif method == "heun":
                x = x + (t1 - t0) * 0.5 * (v0 + velocity(euler, t1))
            else:
                raise ValueError("method must be euler or heun")
        return x.clamp(-1, 1)
