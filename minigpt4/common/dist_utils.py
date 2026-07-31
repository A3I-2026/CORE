

import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile
import random
from typing import List
from tqdm import tqdm

try:
    import decord
    from decord import VideoReader
    decord.bridge.set_bridge("torch")
except Exception:
    decord = None
    VideoReader = None
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.base_dataset import ConcatDataset
MAX_INT = registry.get("MAX_INT")
__all__ = [
    "apply_to_sample",
    "move_to_device",
    "prepare_sample",
    "reorg_datasets_by_split",
    "concat_datasets",
]


class ChainDataset(wds.DataPipeline):
    r"""Dataset for chaining multiple :class:`DataPipeline` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: List[wds.DataPipeline]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prob = []
        self.names = []
        for dataset in self.datasets:
            if hasattr(dataset, 'name'):
                self.names.append(dataset.name)
            else:
                self.names.append('Unknown')
            if hasattr(dataset, 'sample_ratio'):
                self.prob.append(dataset.sample_ratio)
            else:
                self.prob.append(1)
                logging.info("One of the datapipeline doesn't define ratio and set to 1 automatically.")

    def __iter__(self):
        datastreams = [iter(dataset) for dataset in self.datasets]
        while True:
            select_datastream = random.choices(datastreams, weights=self.prob, k=1)[0]
            yield next(select_datastream)


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_device(sample, device):
    def _move(tensor):
        return tensor.to(device)

    return apply_to_sample(_move, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        device = 'cuda'
        try:
            import torch_npu
            if hasattr(torch_npu, 'npu') and torch_npu.npu.is_available():
                device = 'npu'
            elif hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available():
                device = 'npu'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        except Exception:
            if hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available():
                device = 'npu'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        target = device
        if device in ('npu', 'cuda'):
            try:
                import os
                lr_str = os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0'))
                lr = int(lr_str) if str(lr_str).isdigit() else 0
                if device == 'npu':
                    target = f'npu:{lr}'
                else:
                    target = f'cuda:{lr}'
            except Exception:
                target = device
        samples = move_to_device(samples, target)

    # TODO fp16 support

    return samples


def reorg_datasets_by_split(datasets):
    if not isinstance(datasets, dict):
        return {"train": [datasets]}

    keys = list(datasets.keys())
    if len(keys) > 0 and all(k in ("train", "valid", "val", "test") for k in keys):
        reorg = {}
        for split_name, dataset_split in datasets.items():
            if isinstance(dataset_split, list):
                reorg[split_name] = dataset_split
            else:
                reorg[split_name] = [dataset_split]
        return reorg

    reorg_datasets = {}
    for _, dataset in datasets.items():
        if isinstance(dataset, dict):
            for split_name, dataset_split in dataset.items():
                if split_name not in reorg_datasets:
                    reorg_datasets[split_name] = [dataset_split]
                else:
                    reorg_datasets[split_name].append(dataset_split)
        else:
            if "train" not in reorg_datasets:
                reorg_datasets["train"] = [dataset]
            else:
                reorg_datasets["train"].append(dataset)

    return reorg_datasets


def concat_datasets(datasets):

    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            if len(iterable_datasets) > 1:
                chained_datasets = (
                    ChainDataset(iterable_datasets)
                )
            elif len(iterable_datasets) == 1:
                chained_datasets = iterable_datasets[0]
            else:
                chained_datasets = None

            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets
