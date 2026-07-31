"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.models.rec_model import MatrixFactorization
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 

import torch
# Try to import torch_npu to prevent the environment from not loading it automatically
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

if hasattr(torch, 'npu') and torch.npu.is_available():
    device = torch.device("npu")
    print("Detected Device: Huawei Ascend NPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Detected Device: NVIDIA GPU")
else:
    device = torch.device("cpu")
    print("Detected Device: CPU")

import os
from minigpt4.tasks import base_task
import time

def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
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


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', increase=True, patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = increase
        self.reach_count = 0
        self.patience= patience
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    
    # Unified random seed setting (CPU/CUDA/NPU) to ensure reproducibility
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.manual_seed(seed)
    except Exception:
        pass

    # Use relative path matching the project structure
    data_dir = "./datasets/ml-1m/"
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values

    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num =  max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
        else:
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            
    print("user nums:", user_num, "item nums:", item_num)

    mf_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
        }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)

    model = MatrixFactorization(mf_config).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',increase=True,patience=train_config['patience'])
    
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.to(device)
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            users.extend(batch_data[:,0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.to(device)
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc, _, _ = uAUC_me(users, pre, label)

        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return 
    
    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.to(device)
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            loss = criterion(ui_matching,batch_data[:,-1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.to(device)
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc, _, _ = uAUC_me(users, pre, label)

            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.to(device)
                ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            test_uauc, _, _ = uAUC_me(users, pre, label)

            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)

            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                break
            if epoch>100 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 100 epoch but the valid_auc is still less than 0.52")
                break
                
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()

if __name__=='__main__':
    lr_=[1e-3]
    dw_ = [1e-4]
    
    # Dimension alignment: Ensure this strictly matches the LLM configuration file later!
    embedding_size_ = [2048] 
    
    # Use relative path for saving checkpoints
    save_path_dir = "./minigpt4rec-log/"
    os.makedirs(save_path_dir, exist_ok=True)

    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 200,          # 200 epochs are sufficient for the MF model
                    "eval_epoch": 1,
                    "patience": 30,        # Early stopping patience
                    "batch_size": 1024     # Moderate batch size, prevents OOM while maintaining fast training
                }
                print("=========================================")
                print("🚀 Preparing to start MF base training:", train_config)
                
                # Dynamically generate correct filenames to avoid path mismatches
                save_path = f"{save_path_dir}sports_mf_best01_d{embedding_size}.pth"
                print(f"📦 Best weights will be saved at: {save_path}")
                print("=========================================")
                
                run_a_trail(train_config=train_config, 
                            log_file=f, 
                            save_mode=True, 
                            save_file=save_path, 
                            need_train=True, 
                            warm_or_cold=None)
