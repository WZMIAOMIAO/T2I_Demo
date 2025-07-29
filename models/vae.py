from typing import Optional

import torch
from diffusers import AutoencoderKL


class VAEModule:
    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 pretrained_model_name_or_path: str = "stabilityai/sd-vae-ft-mse",
                 scale_factor: Optional[float] = 0.18215):
        self.device = device
        self.dtype = dtype
        self.downsample_ratio = 8
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        self.vae = self.vae.to(dtype=dtype, device=device)
        self.vae.eval()
        # https://github.com/huggingface/diffusers/issues/437
        self.scale_factor = scale_factor

    @torch.inference_mode()
    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(imgs).latent_dist.sample()
        if self.scale_factor is not None:
            latents = latents * self.scale_factor
        return latents

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.scale_factor is not None:
            latents = latents / self.scale_factor
        return self.vae.decode(latents).sample


if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    device = torch.device("cuda:0")
    vae = VAEModule(device)
    pil_image = Image.open("0.png")
    pil_image = pil_image.resize((256, 256))
    pil_image.save("vae_input.png")
    image = torch.from_numpy(np.array(pil_image)).unsqueeze(0).permute([0, 3, 1, 2]) / 255.
    x = (image - 0.5) / 0.5
    res = vae.encode(x.to(device))

    res1 = vae.decode(res)
    res1 = ((res1 * 0.5 + 0.5).permute([0, 2, 3, 1]).squeeze(0) * 255).clip(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(res1).save("vae_output.png")
