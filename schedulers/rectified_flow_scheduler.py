import torch
from diffusers.training_utils import compute_density_for_timestep_sampling


class RFScheduler:
    def __init__(self, time_shift: float, device: torch.device, logit_mean: float = 0., logit_std: float = 1.):
        self.device = device
        self.time_shift = time_shift
        self.logit_mean = logit_mean
        self.logit_std = logit_std

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
