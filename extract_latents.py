import os
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import utils
from dataset import ImageNet256
from models.vae import VAEModule


@torch.inference_mode()
def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(f"cuda:{args.local_rank}")
    save_path = args.save_path

    dataset = ImageNet256(args.dataset_path)
    cls_map = dataset.idx2cls
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=dataset.get_sampler(shuffle=False),
        collate_fn=dataset.collect_fn,
        num_workers=8,
        drop_last=False
    )
    if args.local_rank == 0:
        dataloader = tqdm(dataloader)

    vae = VAEModule(device)

    for data in dataloader:
        imgs = data["img"].to(device)
        label_list = data["label"]
        file_name_list = data["file_name"]

        latents = vae.encode(imgs).cpu()
        for i, (label, file_name) in enumerate(zip(label_list, file_name_list)):
            latent = latents[i].clone()
            save_dir = os.path.join(save_path, cls_map[label])
            os.makedirs(save_dir, exist_ok=True)
            torch.save(latent, os.path.join(save_dir, file_name.replace(".jpg", ".pt")))


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--dataset-path", type=str, required=True)
    argparser.add_argument("--save-path", type=str, required=True)
    argparser.add_argument("--batch-size", type=int, default=32)
    args_ = argparser.parse_args()

    main(args_)
