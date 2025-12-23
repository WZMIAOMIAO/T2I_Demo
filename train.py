import os
import copy
from argparse import ArgumentParser
from pprint import pformat

from easydict import EasyDict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim import AdamW

import utils
from models.dit import get_dit_model, DiT
from models.vae import VAEModule
from dataset import ImageNet256, ImageNet256Latents
from schedulers.rectified_flow_scheduler import RFScheduler


class Trainer:
    def __init__(self, args: EasyDict):
        self.args = args
        utils.init_distributed_mode(self.args)
        log_path = os.path.join(self.args.save_dir, "logs", f"rank_{self.args.rank}.log")
        self.logger = utils.CustomLogger(
            logger_name="trainer",
            rank=self.args.rank,
            local_rank=self.args.local_rank,
            save_file=log_path if self.args.local_rank == 0 else None
        )
        self.logger.info(pformat(self.args), only_local_main_process=True)
        self.device = torch.device(f"cuda:{self.args.local_rank}")
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_bs = self.args.batch_size * self.args.world_size * self.args.accumulate_gradient_steps
        self.update_step = 0
        self.global_step = 0
        self.accumulate_time = 0
        self.accumulate_loss = 0

        if self.args.base_seed:
            seed = self.args.base_seed * self.args.world_size + self.args.rank
            utils.set_seed(seed)
            self.logger.info(f"rank[{self.args.rank}] set seed to {seed}")

        # ------------ #
        # Prepare Data #
        # ------------ #
        dataset_cls = ImageNet256Latents if self.args.use_extract_latents else ImageNet256
        dataset = dataset_cls(args.dataset_path)
        if self.args.rank == 0:
            dataset.dump_cls_dict("cls_map.json")

        self.logger.info(f"find total {len(dataset)} data, {len(dataset.cls2idx)} classes.", only_local_main_process=True)
        self.sampler = dataset.get_sampler(shuffle=True)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            sampler=self.sampler,
            num_workers=self.args.num_workers,
            collate_fn=dataset.collect_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        self.num_iter_per_epoch = len(self.dataloader) // self.args.accumulate_gradient_steps

        # ------------ #
        # Create Model #
        # ------------ #
        if not self.args.use_extract_latents:
            self.vae = VAEModule(pretrained_model_name_or_path=self.args.vae_path, device=self.device)
        else:
            self.logger.info("skip to create vae for saving GPU memory.", only_local_main_process=True)

        dit: DiT = get_dit_model(self.args.dit_name)(class_dropout_prob=self.args.cfg_prob, learn_sigma=False)
        total_params_M = sum([p.numel() for p in dit.parameters()]) / 10e6
        self.logger.info(f"dit params: {total_params_M:.3f}M", only_local_main_process=True)
        if self.args.use_ema:
            self.ema_model = copy.deepcopy(dit)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            self.ema_model.to(self.device)

        dit.to(self.device)
        self.dit = DistributedDataParallel(dit, device_ids=[self.device.index])
        if self.args.use_ema:
            # Ensure EMA is initialized with synced weights
            utils.update_ema(self.ema_model, self.dit.module, decay=0)

        self.optimizer = AdamW(
            params=[p for p in self.dit.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        if self.args.resume:
            self.logger.info(f"resume from {self.args.resume}")
            d_ = torch.load(self.args.resume, map_location="cpu")
            self.dit.module.load_state_dict(d_["model"])
            self.optimizer.load_state_dict(d_["optimizer"])
            self.update_step = d_["update_step"]
            self.global_step = d_["global_step"]
            if self.args.use_ema:
                self.ema_model.load_state_dict(d_["ema"])

        self.scheduler = RFScheduler(self.args.time_shift, device=self.device)
        self.logger.info(f"rank[{self.args.rank}] init complete...")
        dist.barrier()

    def train(self):
        for epoch in range(self.args.epochs):
            self.sampler.set_epoch(epoch)
            for data in self.dataloader:
                self.start_event.record()
                self.global_step += 1
                self.dit.train()
                if self.args.use_extract_latents:
                    latents = data["latent"].to(self.device)
                    labels = torch.tensor(data["label"], device=self.device)
                else:
                    imgs = data["img"].to(self.device)
                    latents = self.vae.encode(imgs)
                    labels = torch.tensor(data["label"], device=self.device)

                loss = self.scheduler.compute_loss(
                    model=self.dit,
                    latents=latents,
                    y=labels
                )
                loss = loss / self.args.accumulate_gradient_steps
                loss.backward()
                self.end_event.record()
                torch.cuda.synchronize()
                elapsed_time = self.start_event.elapsed_time(self.end_event)
                self.accumulate_time += elapsed_time

                loss_detach = loss.detach()
                dist.all_reduce(loss_detach)
                mean_loss = loss_detach / self.args.world_size
                self.accumulate_loss += mean_loss

                if self.global_step % self.args.accumulate_gradient_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.args.use_ema:
                        utils.update_ema(self.ema_model, self.dit.module, self.args.ema_decay)

                    self.update_step += 1
                    msg = f"[Epoch:{epoch}][Step:{self.update_step}] Total bs: {self.total_bs}, "
                    msg += f"Loss: {self.accumulate_loss:.5f}, Time(ms): {self.accumulate_time:.3f}"
                    self.logger.info(msg, only_main_process=True)

                    self.accumulate_loss = 0
                    self.accumulate_time = 0

                    if self.update_step % self.args.save_ckpt_interval == 0:
                        if self.args.rank == 0:
                            checkpoint = {
                                "model": self.dit.module.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                "update_step": self.update_step,
                                "global_step": self.global_step
                            }
                            if self.args.use_ema:
                                checkpoint["ema"] = self.ema_model.state_dict()

                            save_dir = os.path.join(self.args.save_dir, f"checkpoint-{self.update_step}")
                            os.makedirs(save_dir, exist_ok=True)
                            torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))

                        dist.barrier()


if __name__ == '__main__':
    parser = ArgumentParser("T2I trainning Demo.")
    parser.add_argument("--cfg-file", type=str, help="trainging cfg file path.")

    args_ = parser.parse_args()
    trainer = Trainer(EasyDict(utils.read_yaml(args_.cfg_file)))
    trainer.train()
