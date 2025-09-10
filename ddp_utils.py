# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ddp.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-17
# * Version     : 1.0.081701
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import random
import platform
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP             # model

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def seed_torch(seed: int=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def init_env_dist():
    dist.init_process_group(backend="gloo|nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_setup(rank: int, world_size: int):
    """
    function to initialize a distributed process group(1 process/GPU)
    this allows communication among processes

    Args:
        rank (int): a unique process ID(Unique identifier of each process)
        world_size (int): total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    # rank of machine running rank:0 process, assume all GPUs are on the same machine
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    
    # any free port on the machine
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"
    
    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gllo: Facebook Collective Communication Library
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # set deivce
    torch.cuda.set_device(rank)


def destory_dist():
    dist.destroy_process_group()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_env_world_size():
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # world_size = 1
        world_size = torch.cuda.device_count()

    return world_size


def get_env_rank(): 
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    
    return rank


def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)

    return o_all


def wrap_model(model):
    model = DDP(model, device_ids=[dist.get_rank()])

    return model


def print_master_log():
    if is_master():
        # PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        # CUDA version
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
        # CUDA capability
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:
            torch.set_float32_matmul_precision("high")
            logger.info("Uses tensor cores.")
        else:
            logger.info("Tensor cores not supported on this GPU. Using default precision.")




# 测试代码 main 函数
def main():
    ddp_setup(rank=0, world_size=1)
    destory_dist()
    
    if is_dist():
        init_dist()
        init_env_dist()
        ddp_setup(rank=0, world_size=1)
        destory_dist()

    
    logger.info(f"is_dist: {is_dist()}")

    world_size = get_world_size()
    logger.info(f"world_size: {world_size}")

    world_env_size = get_env_world_size()
    logger.info(f"world_env_size: {world_env_size}")

    rank = get_rank()
    logger.info(f"rank: {rank}")

    env_rank = get_env_rank()
    logger.info(f"env_rank: {env_rank}")


    IS_master = is_master()
    logger.info(f"is_master: {IS_master}")
    print_master_log()

    # device = torch.device("cuda", rank)
    # logger.info(f"device: {device}, device.index: {device.index}")

if __name__ == "__main__":
    main()
