import torch
import os

# --- 路径配置 ---
# 1. 身体：一阶段 LoRA 的最佳检查点 (请确保此路径存在)
#path_stage1a = "/tuili/minigpt4rec-log/amazonbook/202604220434/checkpoint_best.pth"

#path_stage1a = "/tuili/minigpt4rec-log/sports_lora/202605070256-epoch157/checkpoint_best.pth"

path_stage1a = "/tuili/minigpt4rec-log/sports_lora/20/20260608081/checkpoint_best.pth" 
# 2. 大脑：软提示预热的最佳检查点 (请确保此路径存在)
#path_stage1b = "/tuili/minigpt4rec-log/book_pod/202604220936/checkpoint_best.pth"

#path_stage1b = "/tuili/minigpt4rec-log/sports_stage2_pod/20260513014/checkpoint_best.pth"

path_stage1b = "/tuili/minigpt4rec-log/sports_stage2_pod/20/20260611014/checkpoint_best.pth"  #无deal
# 3. 输出：二阶段联合训练的启动文件
output_path = "/tuili/minigpt4rec-log/stage2_init_merged_sports20-612.pth"

# --- 执行逻辑 ---
print(f"正在读取 Stage 1a (Base): {path_stage1a}")
if not os.path.exists(path_stage1a):
    print(f"❌ 错误：找不到文件 {path_stage1a}")
    exit(1)
ckpt_a = torch.load(path_stage1a, map_location="cpu")

print(f"正在读取 Stage 1b (Prompt): {path_stage1b}")
if not os.path.exists(path_stage1b):
    print(f"❌ 错误：找不到文件 {path_stage1b}")
    exit(1)
ckpt_b = torch.load(path_stage1b, map_location="cpu")

# 核心合并逻辑
# 以 Stage 1a 为基础 (包含 LoRA, Proj)
merged_state_dict = ckpt_a["model"]

# 将 Stage 1b 中的 Soft Prompt 覆盖/添加进去
if "soft_prompt" in ckpt_b["model"]:
    print("✅ 在 Stage 1b 中发现了 soft_prompt，正在合并...")
    soft_prompt_tensor = ckpt_b["model"]["soft_prompt"]
    merged_state_dict["soft_prompt"] = soft_prompt_tensor
    print(f"   Soft Prompt Shape: {soft_prompt_tensor.shape}")
else:
    print("⚠️ 警告：Stage 1b 中没找到 soft_prompt，请检查文件！")

# 构造新的 Checkpoint 对象
# 既然是新阶段，优化器状态(optimizer)就不要了，让它重新初始化
new_checkpoint = {
    "model": merged_state_dict,
    "config": ckpt_a["config"], # 使用 1a 的配置作为底板
    "epoch": 0 # 重置轮次
}

# 确保输出目录存在
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

torch.save(new_checkpoint, output_path)
print(f"\n🎉 合并成功！已保存至: {output_path}")
print("👉 下一步：请在 stage2_joint_finetune.yaml 中将 resume_ckpt_path 设置为该路径。")