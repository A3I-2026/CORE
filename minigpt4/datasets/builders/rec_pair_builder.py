import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.rec_base_dataset_builder import RecBaseDatasetBuilder
# from minigpt4.datasets.datasets.laion_dataset import LaionDataset
# from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset

from minigpt4.datasets.datasets.rec_datasets import MovielensDataset, MovielensDataset_stage1, AmazonDataset, MoiveOOData, MoiveOOData_sasrec, AmazonOOData, AmazonOOData_sasrec




@registry.register_builder("movie_ood")
class MoiveOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            sip = getattr(self.config, 'selected_indices_path', None)
            if sip is None and hasattr(self.config, 'build_info'):
                try:
                    sip = getattr(self.config.build_info, 'selected_indices_path', None)
                except Exception:
                    sip = None
            if isinstance(sip, str) and os.path.isfile(sip):
                ds = datasets['train']
                if hasattr(ds, 'annotation'):
                    with open(sip, 'r', encoding='utf-8') as f:
                        idxs = [int(line.strip()) for line in f if line.strip()]
                    idxs = [i for i in idxs if 0 <= i < len(ds.annotation)]
                    ds.annotation = ds.annotation.iloc[idxs].reset_index(drop=True)
        except Exception:
            pass
        try:
            sr = getattr(self.config, 'sample_ratio', None)
            if sr is None:
                sr = self.config.get('sample_ratio', None)
            if isinstance(sr, (float, int)) and 0.0 < float(sr) < 1.0:
                ds = datasets['train']
                if hasattr(ds, 'annotation'):
                    ds.annotation = ds.annotation.sample(frac=float(sr), random_state=2023).reset_index(drop=True)
        except Exception:
            pass
        try:
            ds = datasets['train']
            if hasattr(ds, 'annotation'):
                logging.info(f" movie_ood train size after fusion: {len(ds.annotation)}")
                print(f" movie_ood Number of training samples: {len(ds.annotation)}")
        except Exception:
            pass
        try:
            valid_base = os.path.join(storage_path, 'valid_small')
            if not os.path.isfile(valid_base + '_ood2.pkl'):
                valid_base = os.path.join(storage_path, 'valid')
            datasets['valid'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[valid_base])
            datasets['test'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test')])
            if evaluate_only:
                base = os.path.join(storage_path, 'test_warm_cold')
                try:
                    if os.path.isfile(base + '_ood2.pkl'):
                        datasets['test_warm'] = dataset_cls(
                            text_processor=self.text_processors["train"],
                            ann_paths=[base + '=warm'])
                        datasets['test_cold'] = dataset_cls(
                            text_processor=self.text_processors["train"],
                            ann_paths=[base + '=cold'])
                    else:
                        logging.info(f"[Warm/Cold] Skip: {base}_ood2.pkl not found, only build test/valid.")
                except Exception:
                    logging.info("[Warm/Cold] Skip warm/cold split due to exception")
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets


@registry.register_builder("movie_ood_sasrec")
class MoiveOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            sip = getattr(self.config, 'selected_indices_path', None)
            if sip is None and hasattr(self.config, 'build_info'):
                try:
                    sip = getattr(self.config.build_info, 'selected_indices_path', None)
                except Exception:
                    sip = None
            if isinstance(sip, str) and os.path.isfile(sip):
                ds = datasets['train']
                if hasattr(ds, 'annotation'):
                    with open(sip, 'r', encoding='utf-8') as f:
                        idxs = [int(line.strip()) for line in f if line.strip()]
                    idxs = [i for i in idxs if 0 <= i < len(ds.annotation)]
                    ds.annotation = ds.annotation.iloc[idxs].reset_index(drop=True)
        except Exception:
            pass
        try:
            sr = getattr(self.config, 'sample_ratio', None)
            if sr is None:
                sr = self.config.get('sample_ratio', None)
            if isinstance(sr, (float, int)) and 0.0 < float(sr) < 1.0:
                ds = datasets['train']
                if hasattr(ds, 'annotation'):
                    ds.annotation = ds.annotation.sample(frac=float(sr), random_state=2023).reset_index(drop=True)
        except Exception:
            pass
        try:
            ds = datasets['train']
            if hasattr(ds, 'annotation'):
                logging.info(f"movie_ood_sasrec train size after fusion: {len(ds.annotation)}")
                print(f" movie_ood_sasrec Number of training samples: {len(ds.annotation)}")
        except Exception:
            pass
        try:
            valid_base = os.path.join(storage_path, 'valid_small')
            if not os.path.isfile(valid_base + '_ood2.pkl'):
                valid_base = os.path.join(storage_path, 'valid')
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[valid_base])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets



@registry.register_builder("amazon_ood")
class AmazonOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self, evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            sip = getattr(self.config, 'selected_indices_path', None)
            if sip is None and hasattr(self.config, 'build_info'):
                try:
                    sip = getattr(self.config.build_info, 'selected_indices_path', None)
                except Exception:
                    sip = None
            if isinstance(sip, str) and os.path.isfile(sip):
                ds = datasets['train']
                if hasattr(ds, 'annotation'):
                    with open(sip, 'r', encoding='utf-8') as f:
                        idxs = [int(line.strip()) for line in f if line.strip()]
                    idxs = [i for i in idxs if 0 <= i < len(ds.annotation)]
                    ds.annotation = ds.annotation.iloc[idxs].reset_index(drop=True)
        except Exception:
            pass
        try:
            sr = getattr(self.config, 'sample_ratio', None)
            if sr is None:
                sr = self.config.get('sample_ratio', None)
            if isinstance(sr, (float, int)) and 0.0 < float(sr) < 1.0:
                ds = datasets['train']
                if hasattr(ds, 'annotation'):
                    ds.annotation = ds.annotation.sample(frac=float(sr), random_state=2023).reset_index(drop=True)
        except Exception:
            pass
        try:
            ds = datasets['train']
            if hasattr(ds, 'annotation'):
                logging.info(f" amazon_ood train size after fusion: {len(ds.annotation)}")
                print(f"[ amazon_ood Number of training samples: {len(ds.annotation)}")
        except Exception:
            pass
        try:
            valid_base = os.path.join(storage_path, 'valid_small')
            if not os.path.isfile(valid_base + '_ood2.pkl'):
                valid_base = os.path.join(storage_path, 'valid')
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[valid_base])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
            if evaluate_only:
                datasets['test_warm'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test=warm')])

                datasets['test_cold'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test=cold')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets


@registry.register_builder("amazon_ood_sasrec")
class AmazonOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            valid_base = os.path.join(storage_path, 'valid_small')
            if not os.path.isfile(valid_base + '_ood2.pkl'):
                valid_base = os.path.join(storage_path, 'valid')
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[valid_base])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets
