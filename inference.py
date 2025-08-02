import os

import torch
from PIL import Image
from tqdm import tqdm

from models.dit import DiT_L_2
from models.vae import VAEModule
from schedulers.rectified_flow_scheduler import EulerDiscreteSamplingScheduler as SamplingScheduler


@torch.inference_mode()
def main():
    timeshift = 1.6
    num_inference_steps = 50
    cfg_scale = 4.0
    seed = 1234
    use_ema = True
    vae_path = "stabilityai/sd-vae-ft-mse"
    ckpt_path = "outputs/checkpoint-50000/checkpoint.pth"
    save_dir = "visulization/50000iter"
    device = torch.device("cuda:0")

    os.makedirs(save_dir, exist_ok=True)
    dit = DiT_L_2(class_dropout_prob=0.1, learn_sigma=False)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if use_ema:
        dit.load_state_dict(checkpoint["ema"])
    else:
        dit.load_state_dict(checkpoint["model"])
    dit.to(device)
    dit.eval()

    vae = VAEModule(pretrained_model_name_or_path=vae_path, device=device)

    scheduler = SamplingScheduler(
        model=dit,
        num_inference_steps=num_inference_steps,
        time_shift=timeshift,
        device=device
    )
    print(f"inference sigmas: {scheduler.sigmas}")

    # Ensure that the initial latents created by different devices is consistent.
    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    init_latents = torch.randn(
        size=(1, vae.latent_channels, 256 // vae.downsample_ratio, 256 // vae.downsample_ratio),
        generator=generator
    )
    init_latents = init_latents.to(device)

    # refer to cls_map.json
    # 4:acorn, 10:african_chameleon, 12:african_elephant, 20:airship, 25:ambulance
    for label in tqdm([4, 10, 12, 20, 25]):
        latents = init_latents.clone()
        latents = scheduler.generate(
            latents=latents,
            labels=[label],
            cfg_scale=cfg_scale
        )

        img_tensor = vae.decode(latents)
        img_tensor = ((img_tensor * 0.5 + 0.5) * 255).round().clip(0, 255).to(torch.uint8)
        img_np = img_tensor.cpu().permute([0, 2, 3, 1]).squeeze(0).numpy()
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(save_dir, f"label_{label}.png"))


if __name__ == '__main__':
    main()
