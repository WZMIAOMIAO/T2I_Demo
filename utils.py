import os
import yaml
import random
import logging
from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.distributed as dist
import numpy as np
from easydict import EasyDict


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_distributed_mode(args: EasyDict):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.master_addr = os.environ["MASTER_ADDR"]
        args.master_port = os.environ["MASTER_PORT"]
    else:
        raise EnvironmentError('init distributed env failed.')

    torch.cuda.set_device(args.local_rank)
    args.dist_url = f"tcp://{args.master_addr}:{args.master_port}"
    print(f'| distributed init (rank {args.rank}): {args.dist_url}')
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    return data


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if param.requires_grad:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


class CustomLogger:
    def __init__(self,
                 logger_name: str = "custom",
                 rank: int = 0,
                 local_rank: int = 0,
                 level: int = logging.INFO,
                 formatter: Optional[logging.Formatter] = None,
                 save_file: Optional[str] = None):
        """
        初始化自定义日志器

        Args:
            logger_name: 日志器名称
            rank: 全局进程编号
            local_rank: 本地进程编号
            level: 日志级别，默认INFO
            formatter: 日志格式化器，默认使用包含进程信息的格式
        """
        self.rank = rank
        self.local_rank = local_rank
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        # 避免重复添加处理器
        if not self.logger.handlers:
            # 默认格式化器包含进程信息
            if formatter is None:
                formatter = logging.Formatter("[%(levelname)s]%(asctime)s %(filename)s:%(lineno)d %(funcName)s:%(message)s")

            # 输出至控制台
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            if save_file is not None:
                # 输出至文件
                dir_ = os.path.dirname(os.path.abspath(save_file))
                os.makedirs(dir_, exist_ok=True)
                file_handler = logging.FileHandler(filename=save_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def _should_log(self, only_main_process: bool, only_local_main_process: bool) -> bool:
        """判断当前进程是否需要输出日志"""
        if only_local_main_process:
            return self.local_rank == 0
        if only_main_process:
            return self.rank == 0
        return True

    def log(self,
            level: int,
            msg: str,
            *args: Any,
            only_main_process: bool = False,
            only_local_main_process: bool = False,
            **kwargs: Any) -> None:
        """通用日志方法，支持自定义级别"""
        if self._should_log(only_main_process, only_local_main_process):
            self.logger.log(level, msg, *args, stacklevel=3, **kwargs)

    def info(self,
             msg: str,
             *args: Any,
             only_main_process: bool = False,
             only_local_main_process: bool = False,
             **kwargs: Any) -> None:
        """INFO级别日志"""
        self.log(logging.INFO, msg, *args, only_main_process=only_main_process,
                 only_local_main_process=only_local_main_process, **kwargs)

    def debug(self,
              msg: str,
              *args: Any,
              only_main_process: bool = False,
              only_local_main_process: bool = False,
              **kwargs: Any) -> None:
        """DEBUG级别日志"""
        self.log(logging.DEBUG, msg, *args, only_main_process=only_main_process,
                 only_local_main_process=only_local_main_process, **kwargs)

    def warning(self,
                msg: str,
                *args: Any,
                only_main_process: bool = False,
                only_local_main_process: bool = False,
                **kwargs: Any) -> None:
        """WARNING级别日志"""
        self.log(logging.WARNING, msg, *args, only_main_process=only_main_process,
                 only_local_main_process=only_local_main_process, **kwargs)

    def error(self,
              msg: str,
              *args: Any,
              only_main_process: bool = False,
              only_local_main_process: bool = False,
              **kwargs: Any) -> None:
        """ERROR级别日志"""
        self.log(logging.ERROR, msg, *args, only_main_process=only_main_process,
                 only_local_main_process=only_local_main_process, **kwargs)
