import torch


class DDPMScheduler:
    def __init__(self,
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.num_timesteps = num_timesteps

        # Use float64 for accuracy.
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.betas = betas.to(device=device, dtype=torch.float32)
        self.alphas_cumprod = alphas_cumprod.to(device=device, dtype=torch.float32)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=device, dtype=torch.float32)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=device, dtype=torch.float32)

    def compute_loss(self, model, latents: torch.Tensor, **kwargs):
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)

        # sample timesteps
        timesteps = torch.randint(low=0, high=self.num_timesteps, size=(batch_size,), device=self.device)

        # add noise
        noisy_latents = self.sqrt_alphas_cumprod[timesteps].reshape([-1, 1, 1, 1]) * latents + \
            self.sqrt_one_minus_alphas_cumprod[timesteps].reshape([-1, 1, 1, 1]) * noise

        # predict
        kwargs["t"] = timesteps
        pred_noise = model(noisy_latents, **kwargs)

        # compute loss
        err = (pred_noise - noise) ** 2
        loss = err.reshape([batch_size, -1]).mean(dim=1).mean()

        return loss
