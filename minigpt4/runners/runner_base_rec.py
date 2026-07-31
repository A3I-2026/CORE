"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
try:
    from minigpt4.common.dist_utils import (
        download_cached_file,
        get_rank,
        get_world_size,
        is_main_process,
        main_process,
    )
except Exception:
    import timm.models.hub as timm_hub
    import functools
    def download_cached_file(url, check_hash=False, progress=True):
        return timm_hub.download_cached_file(url, check_hash, progress)
    def _is_ddp_init():
        return dist.is_available() and dist.is_initialized()
    def get_world_size():
        return dist.get_world_size() if _is_ddp_init() else 1
    def get_rank():
        return dist.get_rank() if _is_ddp_init() else 0
    def is_main_process():
        return get_rank() == 0
    def main_process(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_main_process():
                return func(*args, **kwargs)
        return wrapper
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from minigpt4.runners.runner_base import RunnerBase



@registry.register_runner("rec_runner_base")
class RecRunnerBase(RunnerBase):
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        print(f"evaluate start  {split_name} epoch {cur_epoch}", flush=True)
        self.model.eval()
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        try:
            self.task.before_evaluation(
                model=model,
                dataset=self.datasets[split_name],
            )
            results = self.task.evaluation(model, data_loader)
            print(f"evaluate over {split_name}，result: {results}", flush=True)

            if results is not None:
                val_log = self.task.after_evaluation(
                    val_result=results,
                    split_name=split_name,
                    epoch=cur_epoch,
                )
                if val_log is not None:
                    try:
                        be = None
                        bm = getattr(self, '_best_metrics', None)
                        if isinstance(bm, dict):
                            be = bm.get('epoch', None)
                    
                        tr_loss = None
                        try:
                            last_tr = getattr(self, '_last_train_stats', None)
                            if isinstance(last_tr, dict):
                                tr_loss = last_tr.get('loss', None)
                        except Exception:
                            tr_loss = None
                        ext = dict(val_log)
                        ext.update({
                            'epoch': int(cur_epoch) if isinstance(cur_epoch, int) else -1,
                            'best_epoch': be,
                            'train_loss': tr_loss,
                        })
                        self.log_stats(stats=ext, split_name=split_name)
                    except Exception:
                        try:
                            self.log_stats(stats=val_log, split_name=split_name)
                        except Exception:
                            pass
                return val_log
            return results
        except Exception as e:
            print(f"evaluate ERROR : {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    @torch.no_grad()
    def eval_epoch_pre(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        self.model.eval()
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
    
    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)
        return results
