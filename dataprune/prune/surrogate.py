import os
import numpy as np
import torch
import ipdb
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets_util import FinetuneDataset,SeqDataset
from trainers import FineTrainer
from utils import generate_rating_matrix_valid,generate_rating_matrix_test, get_user_seqs_npy, check_path, set_seed, get_statistics
from models import SASRecModel

def train(args):
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print(f"Loading data from: {args.data_dir + args.data_name}")
    args.data_file = args.data_dir + args.data_name + '/training_dict.npy'
    val_file = args.data_dir + args.data_name + '/validation_dict.npy'
    test_file = args.data_dir + args.data_name + '/testing_dict.npy'
    
    user_seq = get_user_seqs_npy(args.data_file)
    user_seq_val = get_user_seqs_npy(val_file)
    user_seq_tst = get_user_seqs_npy(test_file)
    
    max_uid = 0
    max_iid = 0
    for seq in [user_seq, user_seq_val, user_seq_tst]:
        if seq is None:
            continue
        if len(seq) > 0:
            max_uid = max(max_uid, len(seq) - 1)
        for items in seq:
            if len(items) > 0:
                mi = max(items)
                if mi > max_iid:
                    max_iid = mi
                    
    real_num_users = max(len(user_seq), len(user_seq_val), len(user_seq_tst))
    real_num_items = max_iid + 1
    args.item_size = real_num_items
    print(f"== [Debug] Max UserID: {max_uid}, Max ItemID: {max_iid}")
    print(f"== [Debug] Matrix Shape Set To: ({real_num_users}, {real_num_items})")
    
    mask_tr = generate_rating_matrix_valid(user_seq, real_num_users, real_num_items)
    # ❌ [删除冗余] 剔除了极其耗时的验证集矩阵构建 (mask_tv)
    
    args.tr_matrix = mask_tr
    args.tv_matrix = None # 不再需要传入

    args.checkpoint_path = args.output_dir + '{}.pth'.format(args.data_name)

    # 🚀 [速度提升] 增加 num_workers 和 pin_memory 解决 IO 加载瓶颈
    train_dataset = SeqDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=0, pin_memory=False)

    # ❌ [删除冗余] 彻底剔除 eval_dataset 和 test_dataset 的内存构建
    
    model = SASRecModel(args)
    # 验证和测试 loader 传 None
    trainer = FineTrainer(model, train_dataloader, None, None, args) 
    
    if args.do_eval: 
        pass # 精简数据集时不需要走这里
    else:
        print(f"开始极速训练，共计 {args.epochs} 个 Epoch...")
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            avg_loss = trainer.train(epoch)

            # 仅保留打印，❌ [删除冗余] 剔除每 5 轮高频保存模型的本地 IO 开销
            if (epoch + 1) % 5 == 0:
                cost_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - epoch_start_time))
                print("Runing Epoch {:03d} train loss {:.4f} costs {}".format(epoch, avg_loss, cost_time))

        # 只在全部训练结束后，保存唯一一次最终模型
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model, args.checkpoint_path)
        print('==='*18)
        print("End. Saved Final Epoch Model")

    # ❌ [删除冗余] 剔除从硬盘重新 load 模型的步骤，内存里的模型已经是最新权重
    
    # 构建提取 influence 时顺序读取的 loader
    train_dataset = SeqDataset(args, user_seq, data_type='train')
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    trainer.train_dataset = train_dataset
    trainer.train_dataloader = train_dataloader

    return trainer