import argparse
import os
import random
import signal
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
try:
    import torch_npu  # ensure Ascend NPU patches are applied before torch operations
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
    import os, datetime, torch
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
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('SLURM_LOCALID', rank))))
        args.rank = rank
        args.world_size = world_size
        args.gpu = local_rank
        args.local_rank = local_rank
        if world_size <= 1:
            args.distributed = False
            return
        args.distributed = True
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=os.environ.get('DIST_URL', f"tcp://127.0.0.1:{os.environ.get('MASTER_PORT','29502')}"),
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def _dealrec_generate_indices(cfg):
    try:
     
        dataset_keys = list(cfg.datasets_cfg.keys())
        if not dataset_keys:
            return None
        dataset_key = dataset_keys[0]
        ds = getattr(cfg.datasets_cfg, dataset_key)
        
        if ds is None:
            return None
            
    
        try:
            sip = getattr(ds, 'selected_indices_path', None)
            if sip is None and hasattr(ds, 'build_info'):
                sip = getattr(ds.build_info, 'selected_indices_path', None)
            if sip and os.path.isfile(sip):
                print(f"🎯  {sip}")
                return sip
        except Exception:
            pass

        try:
            data_dir = ds.path
        except Exception:
            try:
                data_dir = ds.build_info.storage
            except Exception:
                data_dir = None
        if not isinstance(data_dir, str):
            return None
        try:
            df = pd.read_pickle(os.path.join(data_dir, 'train_ood2.pkl')).reset_index(drop=True)
        except Exception:
            return None
        try:
            strict = False
            try:
                strict = bool(getattr(ds.dealrec, 'strict', False))
            except Exception:
                strict = False
            few_shot_size = getattr(ds, 'few_shot_size', None)
            if few_shot_size is None:
                few_shot_size = 1024
            lamda = None
            k_group = None
            surrogate = None
            try:
                lamda = getattr(ds.dealrec, 'lamda', 0.3)
                k_group = getattr(ds.dealrec, 'group_number', 50)
                surrogate = getattr(ds.dealrec, 'surrogate_model', 'SASRec')
            except Exception:
                lamda = 0.3
                k_group = 50
                surrogate = 'SASRec'
        except Exception:
            few_shot_size = 1024
            lamda = 0.3
            k_group = 50
            surrogate = 'SASRec'
            
        if strict:
            try:
                import subprocess, json
                prune_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DEALRec-main', 'code', 'prune')
                data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DEALRec-main', 'data')
                dname = None
                try:
                    dname = getattr(ds.dealrec, 'data_name', None)
                except Exception:
                    dname = None
                if not isinstance(dname, str) or not os.path.isdir(os.path.join(data_root, dname)):
                    print(f" unavailable: {dname}")
                else:
                    cmd = [sys.executable, os.path.join(prune_dir, 'prune.py'),
                           '--data_dir', data_root,
                           '--data_name', dname,
                           '--n_fewshot', str(int(few_shot_size)),
                           '--lamda', str(float(lamda)),
                           '--k', str(int(k_group)),
                           '--hard_prune', str(float(getattr(ds.dealrec, 'hard_prune', 0.1))),
                           '--iteration', str(int(getattr(ds.dealrec, 'iteration', 1))),
                           '--recursion_depth', str(int(getattr(ds.dealrec, 'recursion_depth', 5000)))
                          ]
                    print(f": {' '.join(cmd)}")
                    subprocess.run(cmd, cwd=prune_dir, check=True)
                    import torch
                    pt_path = os.path.join(prune_dir, 'selected', f"{dname}_{int(few_shot_size)}.pt")
                    if os.path.isfile(pt_path):
                        idxs = torch.load(pt_path)
                        out_dir = os.path.join(prune_dir, 'selected')
                        os.makedirs(out_dir, exist_ok=True)
                    
                        out_name = f"{dataset_key}_auto_indices.txt"
                        out_path = os.path.join(out_dir, out_name)
                        
                        with open(out_path, 'w', encoding='utf-8') as f:
                            for idx in idxs:
                                f.write(str(int(idx)) + '\n')
                        print(f"dynamic indices: {out_path}, size: {len(idxs)}")
                        try:
                            setattr(ds, 'selected_indices_path', out_path)
                        except Exception:
                            pass
                        return out_path
            except Exception as e:
                print(f" fail : {e}")
                
        try:
            uid_col = 'uid'
            iid_col = 'iid'
            u = df[uid_col].to_numpy()
            i = df[iid_col].to_numpy()
        except Exception:
            return None
            
        try:
            import numpy as np
            user_counts = np.bincount(u)
            item_counts = np.bincount(i)
            uc = user_counts[u]
            ic = item_counts[i]
            influence = 1.0 / (uc + 1e-6)
            effort = 1.0 / (ic + 1e-6)
            inf = influence
            eff = effort
            inf = (inf - inf.min()) / (inf.max() - inf.min() + 1e-6)
            eff = (eff - eff.min()) / (eff.max() - eff.min() + 1e-6)
            overall = inf + float(lamda) * eff
            order = np.argsort(-overall)
            hard_prune_ratio = 0.1
            keep = order[int(len(order) * hard_prune_ratio):]
            scores = overall[keep]
            s_max = scores.max()
            s_min = scores.min()
            interval = (s_max - s_min) / int(k_group)
            bounds = [min(s_min + interval * _k, s_max) for _k in range(1, int(k_group) + 1)]
            buckets = [[] for _ in range(int(k_group))]
            for idx_pos, s in enumerate(scores):
                for bi, b in enumerate(bounds):
                    if s <= b:
                        buckets[bi].append(int(keep[idx_pos]))
                        break
            selected = []
            m = int(few_shot_size)
            while len(buckets):
                groups = sorted([b for b in buckets if len(b)], key=lambda x: len(x))
                if not len(groups):
                    break
                budget = min(len(groups[0]), max(1, int(m / max(len(groups), 1))))
                if budget <= 0:
                    break
                import random as _r
                chosen = _r.sample(groups[0], budget) if len(groups[0]) >= budget else groups[0]
                selected.extend(chosen)
                buckets = groups[1:]
                m -= len(chosen)
                if m <= 0:
                    break
            if len(selected) > int(few_shot_size):
                selected = selected[:int(few_shot_size)]
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataprune', 'prune', 'selected')
            os.makedirs(out_dir, exist_ok=True)
    
            out_name = f"{dataset_key}_auto_indices.txt"
            out_path = os.path.join(out_dir, out_name)
            
            with open(out_path, 'w', encoding='utf-8') as f:
                for idx in selected:
                    f.write(str(int(idx)) + '\n')
            print(f"dynamic indices: {out_path},size : {len(selected)}")
            try:
                setattr(ds, 'selected_indices_path', out_path)
            except Exception:
                pass
            return out_path
        except Exception:
            return None

    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    
    parser.add_argument("--cfg-path", 
                        default='train_configs/stage1_lora_ml.yaml', 
                        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    print(f" {args.cfg_path}")
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_base"))

    return runner_cls

runner_instance = None
exit_flag = False

def signal_handler(signum, frame):
    global exit_flag
    signal_name = signal.Signals(signum).name
    print(f" {signal_name} ({signum})")
    exit_flag = True
    
  
    if runner_instance is not None:
      
        try:
        
            if hasattr(runner_instance, '_save_checkpoint'):
                current_epoch = getattr(runner_instance, 'current_epoch', 0)
                runner_instance._save_checkpoint(current_epoch, is_best=False)
            elif hasattr(runner_instance, 'save'):
                runner_instance.save(tag='signal_interrupted')
            print(f"CKPT saved success")
        except Exception as e:
            print(f"CKPT saved ERROR: {str(e)}")

def main():
    
    global runner_instance
    

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    
 
    try:
        os.environ["TMPDIR"] = "./tmp"
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        tempfile.tempdir = os.environ.get("TMPDIR")
        print(f"[Temp Directory] TMPDIR={os.environ.get('TMPDIR')}")
    except Exception:
        pass
    try:
       
        default_npu_alloc = "max_split_size_mb:32"
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = os.environ.get("PYTORCH_NPU_ALLOC_CONF", default_npu_alloc)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")
        print(f"[NPU Memory] PYTORCH_NPU_ALLOC_CONF={os.environ.get('PYTORCH_NPU_ALLOC_CONF')}")
    except Exception:
        pass
    try:
        import argparse
        _args = parse_args()
        _dev = getattr(_args, 'device', None)
    except Exception:
        _dev = None
    if str(_dev).lower() == 'npu':
        for _k in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING"]:
            if _k in os.environ:
                os.environ.pop(_k, None)
        print(f"ASCEND_VISIBLE_DEVICES={os.environ.get('ASCEND_VISIBLE_DEVICES')}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    os.environ.setdefault("HCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    try:
        import torch
        if hasattr(torch, 'npu') and torch.npu.is_available():
            _dc = int(torch.npu.device_count())
            for _i in range(_dc):
                try:
                    torch.npu.set_device(_i)
                except Exception as _e:
              
                    raise
    except Exception:
        pass

    init_distributed_mode(cfg.run_cfg)
    try:
        # bind current process to NPU device by LOCAL_RANK after dist init
        _lr = int(os.environ.get('LOCAL_RANK', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0))))
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.set_device(_lr)
            _proc_device = torch.device(f'npu:{_lr}')
        elif torch.cuda.is_available():
            torch.cuda.set_device(_lr)
            _proc_device = torch.device(f'cuda:{_lr}')
        else:
            _proc_device = torch.device('cpu')
        print(f"LOCAL_RANK={_lr}, device={_proc_device}")
    except Exception as _e:
        print(f"fail: {_e}")

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    # cfg.pretty_print()

    try:
        outp = None
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            print("continue construct coreset", flush=True)
        else:
            outp = _dealrec_generate_indices(cfg)
            if isinstance(outp, str):
                print(f"sampling... : {outp}")
    except Exception:
        pass
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    # cfg.model_cfg.get("user_num", "default")
    data_name = list(datasets.keys())[0]
 
    try: #  movie
        data_dir = cfg.datasets_cfg.movie_ood.path
    except: # amazon
        data_dir = cfg.datasets_cfg.amazon_ood.path
    print("data dir:", data_dir)

    train_ = pd.read_pickle(data_dir+"train_ood2.pkl")
    valid_ = pd.read_pickle(data_dir+"valid_ood2.pkl")
    test_ = pd.read_pickle(data_dir+"test_ood2.pkl")
    user_num = max(train_.uid.max(),valid_.uid.max(),test_.uid.max())+1
    item_num = max(train_.iid.max(),valid_.iid.max(),test_.iid.max())+1

    cfg.model_cfg.rec_config.user_num = int(user_num) #int(datasets[data_name]['train'].user_num)  #cfg.model_cfg.get("user_num",)
    cfg.model_cfg.rec_config.item_num = int(item_num) #int(datasets[data_name]['train'].item_num) #cfg.model_cfg.get("item_num", datasets[data_name]['train'].item_num)
    cfg.pretty_print()

    model = task.build_model(cfg)
    try:
        model = model.to(_proc_device)
    except Exception:
        pass
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    
    runner_instance = runner
    
    try:
        # Add key debugging information
        print(f"train config - evaluate_only: {runner.evaluate_only}")
        print(f"max epoch : {runner.max_epoch}")
        print(f"start epoch : {getattr(runner, 'start_epoch', 0)}")
        print(f"need training?: {runner.model_to_betrained()}")
        
        # Check evaluation settings in config
        if hasattr(cfg, 'run_cfg') and hasattr(cfg.run_cfg, 'evaluate'):
            print(f"evaluate signal: {cfg.run_cfg.evaluate}")
        
        runner.train()
        print(f"[Process  {get_rank()}] finished !")
    except KeyboardInterrupt:
        print(f"[Process {get_rank()}] keyboard error, exiting...")
    except Exception as e:
        print(f"[Process {get_rank()}] train ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        # Clean up distributed environment
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
                print(f"[Process {get_rank()}] Distributed process group destroyed")
            except Exception as e:
                print(f"[Process {get_rank()}] Error destroying distributed process group: {str(e)}")
        
        if hasattr(torch, "npu"):
            try:
                torch.npu.empty_cache()
                print(f"[Process {get_rank()}] NPU memory cleared")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Process {get_rank()}] GPU memory cleared")


if __name__ == "__main__":
    # Prioritize setting start method to avoid forkserver triggering connection failures during initialization
    try:
        mp.set_start_method('spawn', force=True)
        print('[Process] Enabled spawn start method')
    except Exception as _e:
        print(f'[Process] Failed to set spawn: {_e}')

    # Register signal handlers again to ensure they take effect
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print("[Main Entry] Signal handlers registered")
    
    try:
        main()
    except Exception as e:
        print(f"[Main Entry] Main function error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        print(f"[Main Entry] Executing final cleanup")
        if runner_instance is not None:
            try:
                print(f"[Main Entry] Attempting to save final checkpoint...")
                if hasattr(runner_instance, '_save_checkpoint'):
                    current_epoch = getattr(runner_instance, 'current_epoch', 0)
                    runner_instance._save_checkpoint(current_epoch, is_best=False)
                print(f"[Main Entry] Final checkpoint saved successfully")
            except Exception as e:
                print(f"[Main Entry] Error saving final checkpoint: {str(e)}")
        print(f"[Main Entry] Program exited")
