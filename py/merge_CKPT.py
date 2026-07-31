import torch
import os

# --- Path Configuration ---
# 1. Stage 1 LoRA checkpoint
path_stage1a = "./minigpt4rec-log/lora/checkpoint_best.pth" 

# 2. Best checkpoint of the soft prompt warm-up
path_stage1b = "./minigpt4rec-log/ml1m_stage2_pod/checkpoint_best.pth"  

# 3. Output: Initial checkpoint for joint training
output_path = "./minigpt4rec-log/stage2_init_merged_lora.pth"

# --- Execution Logic ---
print(f"Reading Stage 1a (Base): {path_stage1a}")
if not os.path.exists(path_stage1a):
    print(f"❌ Error: File not found {path_stage1a}")
    exit(1)
ckpt_a = torch.load(path_stage1a, map_location="cpu")

print(f"Reading Stage 1b (Prompt): {path_stage1b}")
if not os.path.exists(path_stage1b):
    print(f"❌ Error: File not found {path_stage1b}")
    exit(1)
ckpt_b = torch.load(path_stage1b, map_location="cpu")

# Core merging logic
# Base on Stage 1a (contains LoRA, Proj)
merged_state_dict = ckpt_a["model"]

# Overwrite/add Soft Prompt from Stage 1b
if "soft_prompt" in ckpt_b["model"]:
    print("✅ Found soft_prompt in Stage 1b, merging...")
    soft_prompt_tensor = ckpt_b["model"]["soft_prompt"]
    merged_state_dict["soft_prompt"] = soft_prompt_tensor
    print(f"   Soft Prompt Shape: {soft_prompt_tensor.shape}")
else:
    print("⚠️ Warning: soft_prompt not found in Stage 1b, please check the file!")

# Construct the new Checkpoint object
# Since it's a new training stage, discard the optimizer state to force re-initialization
new_checkpoint = {
    "model": merged_state_dict,
    "config": ckpt_a["config"], # Use config from 1a as the template
    "epoch": 0 # Reset epoch
}

# Ensure the output directory exists
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

torch.save(new_checkpoint, output_path)
print(f"\n🎉 Successfully merged! Saved to: {output_path}")
print("👉 Next step: Please set resume_ckpt_path to this path in your stage3.yaml configuration.")
