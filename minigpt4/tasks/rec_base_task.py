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
from minigpt4.common.logger import MetricLogger, SmoothedValue, MetricLogger_auc, SmoothedValue_v2
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from transformers import GenerationConfig
from sklearn.metrics import roc_auc_score,accuracy_score
from minigpt4.tasks.base_task import BaseTask
import time
import numpy as np

def uAUC_me(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user

# Function to gather tensors across processes
def gather_tensor(tensor, dst=0):
    if dist.is_available():
        world_size = dist.get_world_size()
        if world_size > 1:
            if not isinstance(tensor, list):
                tensor = [tensor]

            gathered_tensors = [torch.empty_like(t) for t in tensor]
            dist.gather(tensor, gathered_tensors, dst=dst)

            return gathered_tensors
        else:
            return tensor
    else:
        return tensor

class RecBaseTask(BaseTask):
    def train_step(self, model, samples):
        # Ensure the model is called and returns the correct loss
        # Adjust here according to the model's actual implementation to ensure the loss is correctly computed and backpropagated
        output = model(samples)
        # Ensure output is a dictionary and contains the loss key
        if isinstance(output, dict) and 'loss' in output:
            return output['loss']
        else:
            # If the model does not return a dict or has no loss key, try using output directly as loss
            print(f"[Warning] Model did not return a dictionary containing loss, attempting to use output directly as loss")
            return output
    
    def valid_step(self, model, samples):
        try:
            import torch as _t
            with _t.no_grad():
                outputs = model(samples)
        except Exception:
            outputs = model(samples)
        if isinstance(outputs, dict):
            return outputs
        return {"loss": outputs}

    def before_evaluation(self, model, dataset, **kwargs):
        pass

    def after_evaluation(self, **kwargs):
        val_result = kwargs.get('val_result')
        # Ensure the returned result contains the agg_metrics field, which is required by the Runner
        if val_result is not None and 'agg_metrics' not in val_result:
            # If there is no agg_metrics, create one based on loss
            loss = val_result.get('loss', 0.0)
            # Use negative loss as agg_metrics (assuming smaller loss is better)
            val_result['agg_metrics'] = -loss
            print(f"[Task Evaluation] Added agg_metrics in after_evaluation: {-loss}", flush=True)
        return val_result

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loaders, cuda_enabled=True, max_eval_steps=0, auc_max_samples=0, ndcg_topks=None, hr_topks=None, ndcg_average_mode='all'):
        print("[Task Evaluation] Starting evaluation process", flush=True)
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        auc_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        if hasattr(data_loaders, "loaders"):
            loaders_list = list(data_loaders.loaders)
        else:
            loaders_list = [data_loaders]
        try:
            print_freq = max(1, len(loaders_list[0]) // 10)
        except Exception:
            print_freq = 10

        results = None
        results_loss = []
        
        k = 0
        use_auc = False
        total_loss = 0.0
        num_batches = 0
        import os
        if not isinstance(max_eval_steps, int) or max_eval_steps < 0:
            max_eval_steps = 0
        if max_eval_steps == 0:
            max_eval_steps_env = os.environ.get("EVAL_MAX_STEPS")
            try:
                max_eval_steps = int(max_eval_steps_env) if (max_eval_steps_env and str(max_eval_steps_env).isdigit()) else 0
            except Exception:
                max_eval_steps = 0
        _orig_soft_len = getattr(model, 'soft_prompt_len', None)
        try:
            if isinstance(_orig_soft_len, int) and _orig_soft_len > 48:
                model.soft_prompt_len = 32
        except Exception:
            pass
        for data_loader in loaders_list:
            results_logits = []
            labels = []
            users = []
            seen_any = False
            if hasattr(data_loader, "__len__"):
                iterable = metric_logger.log_every(data_loader, print_freq, header)
            else:
                iterable = data_loader
            for i, samples in enumerate(iterable):
                seen_any = True
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                
                try:
                    import torch as _t
                    with _t.no_grad():
                        eval_output = self.valid_step(model=model, samples=samples)
                        if eval_output is None or 'loss' not in eval_output:
                            print("[Task Evaluation] valid_step did not return loss, attempting to call model directly", flush=True)
                            model_output = model(samples)
                            eval_output = {'loss': model_output.get('loss', 0.0)}
                except Exception as e:
                    print(f"[Task Evaluation] valid_step execution error: {str(e)}", flush=True)
                    try:
                        import torch as _t
                        with _t.no_grad():
                            model_output = model(samples)
                    except Exception:
                        model_output = model(samples)
                    eval_output = {'loss': model_output.get('loss', 0.0)}
                
                # Accumulate loss for stable cross-epoch comparison
                total_loss += float(eval_output['loss'].item())
                num_batches += 1
                if 'logits' in eval_output.keys():
                    use_auc = True
                    import numpy as _np, random as _rand
                    # Lightweight: immediately convert to CPU numpy, and perform reservoir sampling to limit the number of AUC samples
                    user_np = _np.asarray(samples['UserID'].detach().cpu().numpy(), dtype=_np.int64)
                    log_np = _np.nan_to_num(eval_output['logits'].detach().cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0)
                    lab_np = _np.nan_to_num(samples['label'].detach().cpu().numpy(), nan=0.0)
                    for uu, ll, bb in zip(user_np.tolist(), log_np.tolist(), lab_np.tolist()):
                        users.append(uu)
                        results_logits.append(ll)
                        labels.append(bb)
                        if isinstance(auc_max_samples, int) and auc_max_samples > 0:
                            total = len(labels)
                            if total > auc_max_samples:
                                j = _rand.randint(0, total - 1)
                                if j < auc_max_samples:
                                    # Randomly replace a position to maintain a fixed-size buffer
                                    k = _rand.randint(0, auc_max_samples - 1)
                                    labels[k] = labels.pop()
                                    results_logits[k] = results_logits.pop()
                                    users[k] = users.pop()
                                else:
                                    # Directly discard the end to maintain size
                                    labels.pop()
                                    results_logits.pop()
                                    users.pop()
                    logits = torch.nan_to_num(eval_output['logits'], nan=0.0)
                    preds = (logits > 0.5).float()
                    acc = (preds == samples['label'].float()).float().mean()
                    metric_logger.update(acc=acc.item())
                else: 
                    metric_logger.update(acc=0)
                
                metric_logger.update(loss=eval_output['loss'].item())
                if not hasattr(data_loader, "__len__") and (i % print_freq == 0):
                    print(f"{header} [{i}/?] {metric_logger}")
                if max_eval_steps > 0 and (i + 1) >= max_eval_steps:
                    break
            if not seen_any:
                logging.warning("Validation loader is empty, skipping AUC computation.")
                continue
            try:
                import numpy as _np
                logits_np_local = _np.nan_to_num(_np.asarray(results_logits, dtype=_np.float32), nan=0.0, posinf=1.0, neginf=0.0)
                labels_np_local = _np.nan_to_num(_np.asarray(labels, dtype=_np.float32), nan=0.0)
                users_np_local = _np.asarray(users, dtype=_np.int64)
                
                # Distributed aggregation to ensure all ranks use the same batch of evaluation samples for metric computation
                if is_dist_avail_and_initialized() and get_world_size() > 1:
                    import torch as _t
                    dev = _t.device("npu:"+str(_t.npu.current_device())) if hasattr(_t, 'npu') and _t.npu.is_available() else _t.device('cpu')
                    
                    # First aggregate the lengths from each rank to calculate the maximum length
                    n_local = _t.tensor([len(logits_np_local)], device=dev, dtype=_t.int64)
                    sizes = [_t.empty_like(n_local) for _ in range(get_world_size())]
                    dist.all_gather(sizes, n_local)
                    max_len = int(_t.max(_t.stack(sizes)).item())
                    
                    # Construct padded tensors and masks
                    def _pad(arr, length, fill):
                        t = _t.tensor(arr, device=dev)
                        if t.numel() < length:
                            pad = _t.full((length - t.numel(),), fill, device=dev, dtype=t.dtype)
                            t = _t.cat([t, pad], dim=0)
                        return t
                    t_logits = _pad(logits_np_local, max_len, 0.0)
                    t_labels = _pad(labels_np_local, max_len, -1.0)
                    t_users  = _pad(users_np_local,  max_len, -1)
                    mask = _pad([1.0]*len(logits_np_local), max_len, 0.0)
                    
                    # all_gather after fixing length, then concatenate
                    g_logits = [_t.empty_like(t_logits) for _ in range(get_world_size())]
                    g_labels = [_t.empty_like(t_labels) for _ in range(get_world_size())]
                    g_users  = [_t.empty_like(t_users)  for _ in range(get_world_size())]
                    g_masks  = [_t.empty_like(mask)     for _ in range(get_world_size())]
                    dist.all_gather(g_logits, t_logits)
                    dist.all_gather(g_labels, t_labels)
                    dist.all_gather(g_users,  t_users)
                    dist.all_gather(g_masks,  mask)
                    t_logits = _t.cat(g_logits, dim=0).to('cpu')
                    t_labels = _t.cat(g_labels, dim=0).to('cpu')
                    t_users  = _t.cat(g_users,  dim=0).to('cpu')
                    t_masks  = _t.cat(g_masks,  dim=0).to('cpu')
                    
                    # Remove padding
                    keep = t_masks > 0.5
                    logits_np_local = t_logits[keep].numpy()
                    labels_np_local = t_labels[keep].numpy()
                    users_np_local  = t_users[keep].numpy()
            except Exception:
                logits_np_local, labels_np_local, users_np_local = None, None, None
            auc = 0.0
            ndcg_map = {}
            hr_map = {}
            acc_opt = 0.0
            acc_thr = 0.5
            if use_auc and (logits_np_local is not None) and (labels_np_local is not None) and (len(logits_np_local) > 0) and (len(labels_np_local) == len(logits_np_local)):
                try:
                    auc = roc_auc_score(labels_np_local, logits_np_local)
                    import numpy as _np
                    u, inverse, counts = _np.unique(users_np_local, return_inverse=True, return_counts=True)
                    index = _np.argsort(inverse)
                    start = 0
                    aucs = []
                    data = {}
                    for c in counts.tolist():
                        idx = index[start:start+c]
                        if c >= 2:
                            try:
                                aucs.append(roc_auc_score(labels_np_local[idx], logits_np_local[idx]))
                            except Exception:
                                pass
                        s_slice = logits_np_local[idx].tolist()
                        l_slice = labels_np_local[idx].tolist()
                        if len(s_slice) > 0:
                            data.setdefault(int(users_np_local[idx][0]), list())
                            for s, l in zip(s_slice, l_slice):
                                data[int(users_np_local[idx][0])].append((float(s), int(l)))
                        start += c
                    uauc = float(_np.mean(aucs)) if len(aucs) > 0 else 0.0
                    try:
                        ks_ndcg = ndcg_topks if isinstance(ndcg_topks, (list, tuple)) else [10]
                        ks_ndcg = [int(k) for k in ks_ndcg if int(k) > 0]
                        for k in ks_ndcg:
                            ssum, cnt = 0.0, 0
                            for _, arr in data.items():
                                arr_sorted = sorted(arr, key=lambda x: x[0], reverse=True)
                                labs = [1 if int(x[1]) > 0 else 0 for x in arr_sorted][:k]
                                ideal = sorted(labs, reverse=True)
                                dcg = sum((rel / _np.log2(i + 2) for i, rel in enumerate(labs)))
                                idcg = sum((rel / _np.log2(i + 2) for i, rel in enumerate(ideal)))
                                if str(ndcg_average_mode).lower() in ('positives_only','positive_only','valid_only'):
                                    if idcg > 0:
                                        ssum += float(dcg / idcg)
                                        cnt += 1
                                else:
                                    val = float(dcg / idcg) if idcg > 0 else 0.0
                                    ssum += val
                                    cnt += 1
                            if cnt > 0:
                                ndcg_map[f"ndcg@{k}"] = float(ssum / cnt)
                        ks_hr = hr_topks if isinstance(hr_topks, (list, tuple)) else ks_ndcg
                        ks_hr = [int(k) for k in ks_hr if int(k) > 0]
                        for k in ks_hr:
                            hits, users_cnt = 0, 0
                            for _, arr in data.items():
                                arr_sorted = sorted(arr, key=lambda x: x[0], reverse=True)
                                labs = [1 if int(x[1]) > 0 else 0 for x in arr_sorted][:k]
                                if len(labs) > 0:
                                    users_cnt += 1
                                    hits += 1 if any(labs) else 0
                            if users_cnt > 0:
                                hr_map[f"hr@{k}"] = float(hits / users_cnt)
                        try:
                            from sklearn.metrics import roc_curve
                            fpr, tpr, thresholds = roc_curve(labels_np_local, logits_np_local)
                            youden = tpr - fpr
                            best_idx = int(_np.argmax(youden))
                            acc_thr = float(thresholds[best_idx])
                        except Exception:
                            acc_thr = float(_np.median(logits_np_local))
                        acc_opt = float(((logits_np_local > acc_thr).astype(_np.int32) == labels_np_local.astype(_np.int32)).mean())
                    except Exception:
                        ndcg_map = {}
                        hr_map = {}
                        acc_opt = 0.0
                        acc_thr = 0.5
                except Exception as e:
                    logging.error(f"Error computing AUC: {e}")
                    auc = 0.0
                    uauc = 0.0
            try:
                metric_logger.synchronize_between_processes()
            except Exception:
                pass
            if is_main_process():
                logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " ***auc: " + str(auc) + " ***uauc:" + str(uauc) + (" ***ndcg:" + str(ndcg_map) if ndcg_map else "") + (" ***hr:" + str(hr_map) if hr_map else "") + f" ***acc_opt:{acc_opt:.4f} ***acc_thr:{acc_thr:.4f}")
            mean_loss = (total_loss / max(num_batches, 1))
            if use_auc:
                results = {
                    'agg_metrics': float(auc),
                    'auc': float(auc),
                    'acc': float(metric_logger.meters['acc'].global_avg),
                    'loss': float(mean_loss),
                    'uauc': float(uauc),
                }
                results.update(ndcg_map)
                results.update(hr_map)
                results.update({'acc_opt': float(acc_opt), 'acc_thr': float(acc_thr)})
            else:
                results = {
                    'agg_metrics': float(-mean_loss),
                    'auc': 0.0,
                    'acc': 0.0,
                    'loss': float(mean_loss),
                    'uauc': 0.0
                }
            try:
                import numpy as _np
                us = users_np_local if 'users_np_local' in locals() else _np.asarray(users, dtype=_np.int64)
                cand = [16, 32, 64, 128]
                stats = {}
                for S in cand:
                    buckets = us % S
                    counts = _np.bincount(buckets, minlength=S)
                    stats[S] = {
                        'empty_pct': float((counts == 0).sum()) / S,
                        'avg': float(counts.mean()),
                        'median': float(_np.median(counts)),
                        'max': int(counts.max()),
                    }
                if is_main_process():
                    print(f"[Validation User Distribution] soft_prompt_bank candidate bucket statistics: {stats}", flush=True)
            except Exception:
                pass

        try:
            if _orig_soft_len is not None:
                model.soft_prompt_len = _orig_soft_len
        except Exception:
            pass
        if results is None:
            results = {'agg_metrics': 0.0, 'acc': 0.0, 'loss': 0.0, 'uauc': 0.0}
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
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
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

        # Set a more frequent log output frequency to ensure more frequent logging for easier debugging
        progress_output_freq = max(2, log_freq // 5)
        
        # Log training start, clearly showing key parameters
        if is_main_process():
            print(f"[Training Start] Epoch {inner_epoch}, Total iterations: {iters_per_epoch}, log_freq: {log_freq}, progress_output_freq: {progress_output_freq}")
            print(f"[Important] Logs will be output every {progress_output_freq} iterations to ensure training progress is visible")
            logging.info(f"Start training epoch {inner_epoch}, total iterations: {iters_per_epoch}, log frequency: {log_freq}, progress frequency: {progress_output_freq}")
        
        for i in range(iters_per_epoch):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            # Debug: print before and after fetching the first batch to locate blocking points
            if i == 0 and is_main_process():
                print("[Debug] Preparing to fetch the first batch")
            samples = next(data_loader)
            if i == 0 and is_main_process():
                print("[Debug] First batch fetched successfully")

            if i == 0 and is_main_process():
                print("[Debug] Preparing to move samples to CUDA")
            samples = prepare_sample(samples, cuda_enabled=True)
            if i == 0 and is_main_process():
                print("[Debug] Samples moved to CUDA successfully")
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            if i == 0 and is_main_process():
                print("[Debug] Before entering train_step")
                # Print trainable parameters once to confirm optimizer scope
                try:
                    inner_m = getattr(model, 'module', model)
                    print("\n[Debug] Inspecting Trainable Parameters:")
                    for name, param in inner_m.named_parameters():
                        if getattr(param, 'requires_grad', False):
                            try:
                                print(f" -> Trainable: {name} | Shape: {tuple(param.shape)}")
                            except Exception:
                                print(f" -> Trainable: {name}")
                except Exception:
                    pass
            from contextlib import nullcontext
            # Compatible with DDP wrapper, prioritize getting maybe_autocast from module
            maybe_ctx = None
            if scaler is not None:
                try:
                    inner = getattr(model, 'module', model)
                    if hasattr(inner, 'maybe_autocast'):
                        maybe_ctx = inner.maybe_autocast()
                except Exception:
                    maybe_ctx = None
            amp_ctx = maybe_ctx if maybe_ctx is not None else nullcontext()
            with amp_ctx:
                loss = self.train_step(model=model, samples=samples)
            if i == 0 and is_main_process():
                print("[Debug] train_step completed")

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # Check if Soft Prompt has gradients
            if is_main_process():
                try:
                    inner_m = inner if 'inner' in locals() else model
                    if hasattr(inner_m, 'soft_prompt') and inner_m.soft_prompt is not None:
                        if inner_m.soft_prompt.grad is not None:
                            print(f"Soft Prompt Grad Norm: {inner_m.soft_prompt.grad.norm().item():.6f}")
                        else:
                            print("Soft Prompt has NO GRADIENT!")
                except Exception:
                    pass

            # Standardize accumulation interval and end compensation to ensure at least one update per epoch
            effective_accum = min(max(accum_grad_iters, 1), iters_per_epoch)
            if (i + 1) % effective_accum == 0 or (i + 1) == iters_per_epoch:
                # Gradient cleaning: convert NaN/Inf in gradients of all trainable parameters to 0 to prevent the optimizer from being affected by invalid gradients
                try:
                    for p in (inner.parameters() if 'inner' in locals() else model.parameters()):
                        if p.grad is not None:
                            try:
                                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Print gradient norm before zeroing to avoid always seeing 0
                if is_main_process():
                    try:
                        gnorm = 0.0
                        for p in (inner.parameters() if 'inner' in locals() else model.parameters()):
                            if p.grad is not None:
                                try:
                                    gnorm += float(torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0).norm(2).item())
                                except Exception:
                                    pass
                        print(f"[Gradient Norm] Iteration {i}, gnorm: {gnorm:.6f}")
                    except Exception:
                        pass
                if use_amp:
                    try:
                        import torch as _t
                        scaler.unscale_(optimizer)
                        _t.nn.utils.clip_grad_norm_(inner.parameters() if 'inner' in locals() else model.parameters(), max_norm=1.0)
                    except Exception:
                        pass
                    scaler.step(optimizer)
                    scaler.update()                      
                else:    
                    try:
                        import torch as _t
                        _t.nn.utils.clip_grad_norm_(inner.parameters() if 'inner' in locals() else model.parameters(), max_norm=1.0)
                    except Exception:
                        pass
                    optimizer.step()
                optimizer.zero_grad()
                if is_main_process():
                    print(f"[Optimizer Update] Iteration {i}, optimizer updated")

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            # Add more frequent progress log output
            if i % progress_output_freq == 0 or i == iters_per_epoch - 1:
                # Ensure only the main process outputs logs to avoid duplication
                if is_main_process():
                    # Calculate progress percentage
                    progress_percent = (i + 1) / iters_per_epoch * 100
                    # Use both print and logging to ensure visibility in both console and log files
                    progress_str = f"[TRAIN] Epoch {inner_epoch}, Iter {i+1}/{iters_per_epoch} ({progress_percent:.1f}%), Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                    print(progress_str)
                    logging.info(progress_str)
            
            # Keep the original metric log output (remove calls to unavailable displays)
            if i % log_freq == 0:
                if is_main_process():
                    logging.info(f"[METRIC] {header} [{i}/{iters_per_epoch}] {metric_logger}")

        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                pass
            else:
                metric_logger.synchronize_between_processes()
        except Exception:
            pass
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
