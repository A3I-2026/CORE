import argparse
import os
import random
import signal
import sys
import tempfile
import datetime  

import numpy as np
import pandas as pd
import torch
try:
    import torch_npu  
except Exception:
    torch_npu = None
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import multiprocessing as mp

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
try:
    from minigpt4.common.dist_utils import get_rank, init_distributed_mode
except Exception:
    import torch.distributed as dist
    import os, torch
    def get_rank():
        return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    def init_distributed_mode(args):
        if not hasattr(args, 'dist_backend'):
            args.dist_backend = "hccl" if hasattr(torch, 'npu') and torch.npu.is_available() else "nccl"
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29502')
        os.environ.setdefault('HCCL_SOCKET_IFNAME', 'lo')
        os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
        rank = int(os.environ.get('RANK', os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('SLURM_PROCID', 0))))
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1))))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0))))
        args.rank = rank
        args.world_size = world_size
        args.gpu = local_rank
        dist.init_process_group(backend=args.dist_backend, init_method='env://', timeout=datetime.timedelta(hours=2))
        
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.set_device(args.gpu)
        else:
            torch.cuda.set_device(args.gpu)
        dist.barrier()
        return

from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def now():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")

runner_instance = None

def signal_handler(sig, frame):
    print(f'\n[进程 {get_rank()}] 接收到中断信号 ({sig})，正在优雅退出...')
    
    if runner_instance is not None:
        try:
            print(f"[进程 {get_rank()}] 尝试保存检查点...")
            if hasattr(runner_instance, '_save_checkpoint'):
                runner_instance._save_checkpoint(runner_instance.inner_epoch, is_best=False)
            print(f"[进程 {get_rank()}] 检查点保存完成")
        except Exception as e:
            pass
            
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print(f"[进程 {get_rank()}] 分布式进程组已销毁")
        except Exception as e:
            pass
            
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config",
    )
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    global runner_instance
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    #init_distributed_mode(args)

    args.gpu = 0
    args.rank = 0
    args.world_size = 1

    setup_seeds(cfg)
    setup_logger()


    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    
    data_name = list(datasets.keys())[0]

    try:
        dataset_cfg = getattr(cfg.datasets_cfg, data_name)
        if hasattr(dataset_cfg, 'build_info') and hasattr(dataset_cfg.build_info, 'storage'):
            data_dir = dataset_cfg.build_info.storage
        elif hasattr(dataset_cfg, 'path'):
            data_dir = dataset_cfg.path
        else:
            raise ValueError("No path found in config")
    except Exception as e:
        print(f"⚠️ 动态读取路径失败，强行指向 ml-1m 真实物理路径...")
        data_dir = "/root/CoLLM-main/CoLLM/collm-datasets/ml-1m/"
        
    train_ = pd.read_pickle(os.path.join(data_dir, "train_ood2.pkl"))
    valid_ = pd.read_pickle(os.path.join(data_dir, "valid_ood2.pkl"))
    test_ =  pd.read_pickle(os.path.join(data_dir, "test_ood2.pkl"))
    
    # === 核心修复：将 numpy.int64 强制转换为 Python 原生 int ===
    user_num = int(max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1)
    item_num = int(max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1)
    print("user_num,item_num : ", user_num, item_num)

    cfg.model_cfg.user_num = user_num
    cfg.model_cfg.item_num = item_num
    cfg.model_cfg.rec_config.user_num = user_num
    cfg.model_cfg.rec_config.item_num = item_num
    # ==========================================================
    
    model = task.build_model(cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    
    runner_instance = runner
    runner.train()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print('[进程] 启用 spawn 启动方式')
    except Exception as _e:
        print(f'[进程] 设置 spawn 失败: {_e}')

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print("[主入口] 信号处理器已注册")
    
    try:
        main()
    except Exception as e:
        print(f"[主入口] 主函数出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[主入口] 执行最后的清理工作")
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                pass
        if hasattr(torch, "npu"):
            try:
                torch.npu.empty_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
