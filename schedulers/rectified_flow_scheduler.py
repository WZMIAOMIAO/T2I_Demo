from typing import Optional, List

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling


class RFScheduler:
    def __init__(self, time_shift: float, device: torch.device, logit_mean: float = 0., logit_std: float = 1.):
        self.device = device
        self.time_shift = time_shift
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        self.num_inference_steps = None
        self.inference_sigmas = None

    def set_inference_params(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        sigmas = torch.linspace(1, 0, steps=num_inference_steps + 1, dtype=torch.float32, device=self.device)
        self.inference_sigmas = self.time_shift * sigmas / (1 + (self.time_shift - 1) * sigmas)

    def step(self,
             current_sigmas: torch.Tensor,
             next_sigmas: torch.Tensor,
             velocity: torch.Tensor,
             latents: torch.Tensor):
        dtype = latents.dtype
        latents = latents.to(torch.float32)
        latents = latents + (next_sigmas - current_sigmas).reshape([-1, 1, 1, 1]) * velocity
        return latents.to(dtype)

    @torch.inference_mode()
    def generate(self,
                 model,
                 latents: torch.Tensor,
                 labels: List[int],
                 cfg_scale: Optional[float] = None):
        assert self.num_inference_steps is not None
        assert self.inference_sigmas is not None

        for step in range(self.num_inference_steps):
            current_sigma = self.inference_sigmas[step].reshape([1])
            next_sigma = self.inference_sigmas[step + 1].reshape([1])

            pred = model(
                x=latents,
                t=current_sigma * 1000,
                y=torch.tensor(labels, dtype=torch.int64, device=self.device)
            )

            if cfg_scale is not None:
                uncond_pred = model(
                    x=latents,
                    t=current_sigma * 1000,
                    # label 1000 means empty condition
                    y=torch.tensor([1000] * len(labels), dtype=torch.int64, device=self.device)
                )

                # Classifier-Free Guidence
                pred = uncond_pred + cfg_scale * (pred - uncond_pred)

            latents = self.step(current_sigma, next_sigma, pred, latents)

        return latents

    def compute_loss(self, model, latents: torch.Tensor, **kwargs):
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)

        # sample timesteps, refer SD3 paper 5.1.1
        sigmas = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            device=self.device
        )

        # shift sigmas, refer SD3 paper 5.3.2
        sigmas = self.time_shift * sigmas / (1 + (self.time_shift - 1) * sigmas)
        w = sigmas.reshape([batch_size, 1, 1, 1])

        # Add noise
        noisy_latents = w * noise + (1 - w) * latents

        # flow matching target
        target = noise - latents

        # predict
        kwargs["t"] = sigmas * 1000
        velocity = model(noisy_latents, **kwargs)

        # compute loss
        err = (velocity - target) ** 2
        loss = err.reshape([batch_size, -1]).mean(dim=1).mean()

        return loss
