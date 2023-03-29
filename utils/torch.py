import os
import torch


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(use=False, mode='single', local_rank=0, nprocs=1):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    
    if use:
        use_gpu = True
        
        if mode == 'single':
            ev = os.environ.get("CUDA_VISIBLE_DEVICES")
            if ev != None:
                gpu_list = ev.split(',')
                gpu_id = int(gpu_list[0])
        elif mode == 'torch.distributed':
            dist_rank = local_rank
            world_size = nprocs
            gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES").split(',')
            gpu_id = int(gpu_list[dist_rank]) 
        elif mode == 'slurm':
            gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
            dist_rank = int(os.environ.get("SLURM_PROCID", 0))
            world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
        distributed = world_size > 1
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        