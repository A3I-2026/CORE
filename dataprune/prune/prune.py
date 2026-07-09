from surrogate import train
from utils import get_args
from influence_score import get_influence_score
from effort_score import get_effort_score

import torch
import math
import random
import numpy as np
import os

if __name__ == '__main__':
    args = get_args()
    
    # ==========================================
    # 1. 训练 SASRec 代理模型 (初筛第一级)
    # ==========================================
    print("🚀 [Step 1] 正在训练 SASRec 并计算 Influence Score...")
    trainer = train(args)
    influence = get_influence_score(args, trainer)
    raw_user_seq = trainer.train_dataset.user_seq 
    
    del trainer
    torch.cuda.empty_cache()  

    # ==========================================
    # 🌟 终极提速：SASRec 智能漏斗初筛
    # ==========================================
    print("\n🚀 [提速优化] 正在启动智能漏斗，拦截低潜力用户...")
    influence_tensor = torch.tensor(influence)
    
    # 仅允许排名前 30,000 的高潜力用户进入极其耗时的大模型算分阶段
    max_llm_candidates = 30000 
    
    active_mask = torch.tensor([len(seq) >= 5 for seq in raw_user_seq])
    masked_influence = torch.where(active_mask, influence_tensor, torch.tensor(float('-inf')))
    
    actual_candidates = min(max_llm_candidates, int(active_mask.sum().item()))
    top_k_values, top_k_indices = torch.topk(masked_influence, actual_candidates)
    top_k_indices_list = top_k_indices.tolist()
    
    candidate_seqs = [raw_user_seq[i] for i in top_k_indices_list]
    
    print(f"📊 原始活跃用户: {int(active_mask.sum().item())}")
    print(f"⚡ 漏斗拦截成功！仅将最具价值的 {actual_candidates} 名候选人送入 Vicuna-7B。")
    
    # ==========================================
    # 2. 加载 Vicuna-7B 计算 Effort Score (仅算 3 万人)
    # ==========================================
    print(f"\n🚀 [Step 2] 正在加载 {args.base_model} 计算 Effort Score...")
    candidate_effort = get_effort_score(args, user_seq=candidate_seqs)
    
    # 将候选人的分数还原，被拦截的用户默认 0 分
    effort = torch.zeros(len(raw_user_seq))
    if isinstance(candidate_effort, list):
        candidate_effort = torch.tensor(candidate_effort)
        
    for i, original_idx in enumerate(top_k_indices_list):
        if i < len(candidate_effort):
            effort[original_idx] = float(candidate_effort[i])
            
    # ==========================================
    # 3. 强制对齐与双重归一化 
    # ==========================================
    print("\n🚀 [Step 3] 正在融合双擎分数并执行 DEALRec 自适应剪枝...")
    if isinstance(influence, list):
        influence = torch.tensor(influence)
        
    min_len = min(len(influence), len(effort))
    influence = influence[:min_len]
    effort = effort[:min_len]
    
    inf_range = torch.max(influence) - torch.min(influence)
    eff_range = torch.max(effort) - torch.min(effort)
    influence_norm = (influence - torch.min(influence)) / (inf_range if inf_range > 0 else 1e-9)
    effort_norm = (effort - torch.min(effort)) / (eff_range if eff_range > 0 else 1e-9)
    
    overall = influence_norm + args.lamda * effort_norm
    scores_sorted, indices = torch.sort(overall, descending=True)

    n_prune = math.floor(0.01 * len(scores_sorted))
    scores_sorted = scores_sorted[n_prune:]
    indices = indices[n_prune:]
    total_candidates = len(scores_sorted)

    overall_np = scores_sorted.cpu().numpy()
    indices_np = indices.cpu().numpy()

    # ==========================================
    # 4. 高覆盖率自适应提取
    # ==========================================
    target_mass_ratio = 0.75 
    cumulative_scores = np.cumsum(overall_np)
    dynamic_top_idx = np.searchsorted(cumulative_scores, cumulative_scores[-1] * target_mass_ratio)
    
    min_core = int(total_candidates * 0.30)
    max_core = int(total_candidates * 0.60)
    dynamic_top_idx = max(min_core, min(dynamic_top_idx, max_core))
    coreset = indices_np[:dynamic_top_idx].tolist()

    remaining_scores = overall_np[dynamic_top_idx:]
    remaining_indices = indices_np[dynamic_top_idx:]

    if len(remaining_scores) > 0:
        bins = np.linspace(np.min(remaining_scores), np.max(remaining_scores), args.k + 1)
        digitized = np.digitize(remaining_scores, bins)

        for i in range(1, args.k + 1):
            mask = (digitized == i)
            bin_indices = remaining_indices[mask]
            if len(bin_indices) > 0:
                cv = np.std(remaining_scores[mask]) / (np.mean(remaining_scores[mask]) + 1e-9)
                sample_size = math.ceil(len(bin_indices) * (0.10 + min(cv / 0.5, 1.0) * 0.20))
                sample_size = min(sample_size, len(bin_indices))
                coreset.extend(random.sample(bin_indices.tolist(), sample_size))

    # ==========================================
    # 5. 保存结果 (支持自定义命名)
    # ==========================================
    final_size = len(coreset)
    print(f"\n==================================================")
    print(f"✅ [最终结果] 适合 CORE 模型微调的自适应精简完成！")
    print(f"📊 原始数据量: {min_len} | 精简后数据量: {final_size} | 保留率: {final_size/min_len * 100:.2f}%")
    print(f"==================================================")

    os.makedirs("selected", exist_ok=True)
    
    # 💡 核心修改：读取命令行传入的 log_name 作为文件后缀
    custom_suffix = getattr(args, 'log_name', 'adaptive_coreset')
    save_path = f"selected/{args.data_name}_{custom_suffix}.pt"
    
    torch.save(coreset, save_path)
    print(f"📁 核心数据索引已成功保存至: {save_path}")
