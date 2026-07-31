"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import inspect

import torch
import torch.distributed as dist
try:
    from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
except Exception:
    def _is_ddp_init():
        return dist.is_available() and dist.is_initialized()
    def get_world_size():
        return dist.get_world_size() if _is_ddp_init() else 1
    def get_rank():
        return dist.get_rank() if _is_ddp_init() else 0
    def is_main_process():
        return get_rank() == 0
    def is_dist_avail_and_initialized():
        return _is_ddp_init()
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
try:
    from minigpt4.datasets.data_utils import prepare_sample
except Exception:
    import torch
    from minigpt4.datasets.data_utils import apply_to_sample, move_to_device
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
            samples = move_to_device(samples, device)
        return samples


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        model_kwargs = dict(model_config)
        if 'arch' in model_kwargs:
            del model_kwargs['arch']
        if 'model_type' in model_kwargs:
            del model_kwargs['model_type']
        if 'proj_mid_times' in model_kwargs and 'proj_mid' not in model_kwargs:
            model_kwargs['proj_mid'] = model_kwargs.pop('proj_mid_times')

        sig = inspect.signature(model_cls.__init__)
        allowed = {k for k in sig.parameters.keys() if k != 'self'}
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}

        return model_cls(**model_kwargs)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        evaluate_only = cfg.run_cfg.evaluate

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(evaluate_only=evaluate_only)

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        save_batch_size = 1000
        results = []
        temp_results = []

        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            with torch.no_grad():
                eval_output = self.valid_step(model=model, samples=samples)
                
            if isinstance(eval_output, list):
                temp_results.extend(eval_output)
            
            if len(temp_results) >= save_batch_size:
                results.extend(temp_results)
                temp_results.clear()
                
                import gc, torch
                gc.collect()
                try:
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif torch.cuda.is_available():
                        try:
                            if hasattr(torch, 'npu') and torch.npu.is_available():
                                torch.npu.empty_cache()
                            elif torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                except Exception:
                    pass
                print(f"[Memory Optimization] Processed {i+1} batches, cleared temporary result memory.")
            
            # Memory optimization: Clear cache periodically
            if (i + 1) % (print_freq * 2) == 0:
                try:
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"[Memory Optimization] Processed {i+1} batches, cleared CUDA cache.")
        
        if temp_results:
            results.extend(temp_results)
            temp_results.clear()
            try:
                import torch
                try:
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

        if is_dist_avail_and_initialized():
            dist.barrier()
            # Additional memory cleanup in distributed environment
            try:
                try:
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

        print(f"[Memory Optimization] Evaluation completed. Processed {len(results)} results in total.")
        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=True,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=True,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=True,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        # Set model to training mode
        model.train()
        
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        progress_freq = max(1, log_freq // 5)
        
        # Track loss changes for debugging
        prev_loss = None
        valid_steps = 0
        
        # Memory monitoring and cleanup parameters
        clean_freq = max(1, accum_grad_iters // 2)
        
        for i in range(iters_per_epoch):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            # Periodically print progress info
            if i % progress_freq == 0:
                print(f"[Training Progress] Epoch {inner_epoch}, Iteration {i}/{iters_per_epoch}")

            try:
                samples = next(data_loader)
            except StopIteration:
                # Memory optimization: Reinitialize data loader upon StopIteration
                print(f"[DataLoader] Encountered StopIteration, reinitializing data loader.")
                break

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            # Strict context management for memory optimization
            from contextlib import nullcontext
            amp_ctx = (model.maybe_autocast(torch.float16) if (use_amp and hasattr(model, 'maybe_autocast')) else nullcontext())
            with amp_ctx:
                with torch.set_grad_enabled(True):
                    loss = self.train_step(model=model, samples=samples)
                    
                    # Ensure loss value is valid
                    if loss is None or torch.isnan(loss):
                        print(f"[Warning] Invalid loss value at iteration {i}: {loss}")
                        # Skip backward pass to avoid gradient graph fragmentation
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        continue

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Check gradient updates
            if i % progress_freq == 0:
                # Check gradient norm of the first few parameter groups for memory optimization
                param_norm = 0
                for param in list(model.parameters())[:10]:  
                    if param.grad is not None:
                        param_norm += param.grad.data.norm(2).item()
                param_norm = param_norm ** 0.5 if param_norm > 0 else 0
                print(f"[Gradient Check] Iteration {i}, gradient norm: {param_norm:.6f}")

            # Standardize update frequency
            effective_accum = min(max(accum_grad_iters, 1), iters_per_epoch)
            
            # Update gradients every effective_accum iterations with final compensation
            if (i + 1) % effective_accum == 0 or (i + 1) == iters_per_epoch:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    # Gradient clipping to prevent NaN
                    try:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    except Exception:
                        pass
                    optimizer.step()
                optimizer.zero_grad()
                print(f"[Optimizer Update] Iteration {i}, optimizer updated.")
                
                # Immediate memory cleanup post-update
                try:
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                import gc
                gc.collect()
            
            # More frequent memory cache cleanups
            elif (i + 1) % clean_freq == 0:
                try:
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # Record loss value to scalar, releasing the computation graph
            current_loss = loss.item()  
            if prev_loss is not None and i % progress_freq == 0:
                loss_change = current_loss - prev_loss
                print(f"[Loss Change] Iteration {i}, Loss: {current_loss:.6f}, Change: {loss_change:.6f}")
            prev_loss = current_loss
            
            metric_logger.update(loss=current_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=current_lr)
            valid_steps += 1
            
            # Print detailed logs periodically
            if i % log_freq == 0:
                print(f"[Detailed Log] Epoch {inner_epoch}, Iteration {i}, Loss: {current_loss:.6f}, LR: {current_lr:.6f}")
                torch.cuda.empty_cache()
            
            # Delete temporary variables
            del loss, current_loss
            
            # Cache cleanup at end of iteration
            try:
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            metric_logger.synchronize_between_processes()
        except Exception:
            pass
            
        if valid_steps > 0:
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))
            return {
                k: "{:.3f}".format(meter.global_avg)
                for k, meter in metric_logger.meters.items()
            }
        else:
            logging.warning("No valid training steps in this epoch; skipping averaged stats")
            return {"lr": optimizer.param_groups[0]["lr"], "loss": float(metric_logger.meters['loss'].global_avg) if 'loss' in metric_logger.meters else float('nan')}

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
