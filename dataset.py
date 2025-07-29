import os
import json
from typing import List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DistributedSampler
from torchvision.transforms import transforms
from PIL import Image


@dataclass
class DataItem:
    cls_name: str
    file_name: str
    file_path: str


class ImageNet256(Dataset):
    def __init__(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        cls_names = sorted(os.listdir(dataset_path))
        self.cls2idx = dict([(cls_name, i) for i, cls_name in enumerate(cls_names)])
        self.idx2cls = dict([(i, cls_name) for i, cls_name in enumerate(cls_names)])
        self.data_list: List[DataItem] = []
        for cls_name in cls_names:
            cls_dir = os.path.join(dataset_path, cls_name)
            for file_name in sorted(os.listdir(cls_dir)):
                file_path = os.path.join(cls_dir, file_name)
                self.data_list.append(DataItem(cls_name, file_name, file_path))

        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            ]
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.data_list[index]
        label = self.cls2idx[data_item.cls_name]
        pil_img = Image.open(data_item.file_path).convert("RGB")
        img_tensor: torch.Tensor = self.transforms(pil_img)
        return {"img": img_tensor, "label": label, "cls_name": data_item.cls_name, "file_name": data_item.file_name}

    def get_sampler(self, shuffle: bool = True):
        return DistributedSampler(self, shuffle=shuffle)

    def dump_cls_dict(self, save_path: str, reverse: bool = True):
        with open(save_path, mode="wt", encoding="utf-8") as f:
            json.dump(self.idx2cls if reverse else self.cls2idx, f, indent=4)

    @staticmethod
    def collect_fn(data_list):
        keys = list(data_list[0].keys())
        res = {}
        for k in keys:
            if k == "img":
                imgs = torch.stack([_["img"] for _ in data_list], dim=0)
                res[k] = imgs
            else:
                v_list = [_[k] for _ in data_list]
                res[k] = v_list

        return res


class ImageNet256Latents(Dataset):
    def __init__(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        cls_names = sorted(os.listdir(dataset_path))
        self.cls2idx = dict([(cls_name, i) for i, cls_name in enumerate(cls_names)])
        self.idx2cls = dict([(i, cls_name) for i, cls_name in enumerate(cls_names)])
        self.data_list: List[DataItem] = []
        for cls_name in cls_names:
            cls_dir = os.path.join(dataset_path, cls_name)
            for file_name in sorted(os.listdir(cls_dir)):
                file_path = os.path.join(cls_dir, file_name)
                self.data_list.append(DataItem(cls_name, file_name, file_path))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_item = self.data_list[index]
        label = self.cls2idx[data_item.cls_name]
        latent_tensor: torch.Tensor = torch.load(data_item.file_path)
        return {"latent": latent_tensor, "label": label, "cls_name": data_item.cls_name, "file_name": data_item.file_name}

    def get_sampler(self, shuffle: bool = True):
        return DistributedSampler(self, shuffle=shuffle)

    def dump_cls_dict(self, save_path: str, reverse: bool = True):
        with open(save_path, mode="wt", encoding="utf-8") as f:
            json.dump(self.idx2cls if reverse else self.cls2idx, f, indent=4)

    @staticmethod
    def collect_fn(data_list):
        keys = list(data_list[0].keys())
        res = {}
        for k in keys:
            if k == "latent":
                latents = torch.stack([_["latent"] for _ in data_list], dim=0)
                res[k] = latents
            else:
                v_list = [_[k] for _ in data_list]
                res[k] = v_list

        return res
