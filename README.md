
# CORE: Collaborative Optimization of Recommendation Effectiveness via Multi-fusion Training with Structural Embedding-based LLMs

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **"[Your Paper Title Here]"**[cite: 12]. CORE is an efficient three-stage training framework for LLM-based recommendation systems[cite: 12].

Quick Start

1. Environment Setup


git clone [https://github.com/zzz2025-ai/CORE.git] && cd CORE
conda create -n core python=3.9 -y && conda activate core

# Adjust the CUDA version (cu118) as needed
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt


2. Data Preparation

We support MovieLens-1M, Amazon-Books, and Amazon-Sports. Download the raw datasets and place them in the paths configured in `/CORE-main/datasets/`.

 3. Training Pipeline

CORE follows a systematic three-stage paradigm (using amazon-books as an example):

#### Stage 1: Dual-Score Data Pruning

Extracts a high-value coreset by computing Influence & Effort scores via a surrogate model and a Zero-Shot LLM. The code for data pruning, including the score calculation and the coverage-enhanced sample selection is in `./code/prune/`. You can prune the data by running:

python -u prune.py --data_name=$1 --model_name=$2 --lamda=$3 --k=$4 --log_name=$5 --gpu_id=$6

or use `prune.sh`:

sh prune.sh <data_name> <surrogate_model_name> <lamda> <group_number> <log_name> <gpu_id>


**Example : Prune the data on Movielens-1M

cd ./CORE-main/prune/
sh prune.sh ml-1m SASRec 0.3 50 log 0


* The selected samples' indices will be saved in `./dataprune/prune/selected` folder.


* The explanation of hyper-parameters can be found in `./dataprune/prune/utils.py`.



🌟 **Note:** The surrogate model implemented here is SASRec.

#### Stage 2: LoRA Optimization & Soft Prompt Distillation

On the extracted coreset, we initialize and train the LLM's LoRA parameters using discrete prompts. Simultaneously, we perform Soft Prompt Distillation via MSE loss, forcing continuous soft prompts to absorb the rich semantic information from the discrete text.

python ./py/train_sasrec.py --cfg-path train_configs/stage1_lora_book.yaml


 ⚠️ **Important:** After the LoRA training is complete, a model checkpoint (`.pth` file) will be saved. Before proceeding to the distillation step, you must copy the path of this generated `.pth` file and update the corresponding checkpoint parameter inside `train_configs/stage2_pod_book.yaml`.


python ./py/train_sasrec.py --cfg-path train_configs/stage2_pod_book.yaml


#### Stage 3: Semantic-Collaborative Fusion

⚠️ **Important:** After both the LoRA and Soft Prompt Distillation (POD) training phases are complete, you must merge their respective `.pth` files into a new unified checkpoint using the `merge_CKPT.py` script. Before proceeding to Stage 3, copy the path of this newly merged `.pth` file and update the corresponding checkpoint parameter inside `train_configs/stage3_cie_dis_book.yaml`.

python ./py/merge_CKPT.py
python ./py/train.py --cfg-path train_configs/stage3_cie_dis_book.yaml






