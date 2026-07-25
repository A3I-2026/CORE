import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
# from peft import PeftModel  # 注释掉不需要的 PEFT

import utils

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning / Effort Score Calculation."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, user_seq=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        instruction = "Could you tell me what item the user will interact with next?"
        
        input_ids = []
        labels = []
        
        for idx in range(len(user_seq)):
            input_seq = user_seq[idx]

            if len(input_seq) > 64:
                input_seq = input_seq[-64:]
                
            input_str = "The user has interacted with the following items: " + ' '.join(str(s) for s in input_seq[:-1])
            target_str = str(input_seq[-1])

            source = prompt_input.format_map({'instruction': instruction, 'input': input_str})
            target = f"{target_str}{tokenizer.eos_token}"

            tokenized_source = tokenizer(
                source,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            tokenized_target = tokenizer(
                source + target,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            target_labels = copy.deepcopy(tokenized_target["input_ids"][0])
            source_len = tokenized_source["input_ids"][0].ne(tokenizer.pad_token_id).sum().item()
            target_labels[:source_len] = IGNORE_INDEX
            
            input_ids.append(tokenized_target["input_ids"][0])
            labels.append(target_labels)

        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def get_effort_score(args, user_seq=None):
    """
    计算 Effort Score
    """
    print("🚀 正在加载基础大语言模型 (LLM) 进行 Effort Score 评估...")
    
    # 恢复原版 LlamaForCausalLM 导入，避免模型加载报错
    model = transformers.LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,  # 开启半精度，降低显存
    )
    
    # 将模型手动推至设备 (适配你的 NPU 环境)
    device = torch.device("npu:0" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda:0")
    model = model.to(device)



    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.base_model,
        model_max_length=512, # 全局安全长度限制
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SupervisedDataset(data_path=None, tokenizer=tokenizer, user_seq=user_seq)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # 关闭多余日志，提高计算速度
    training_args = transformers.TrainingArguments(
        output_dir="./tmp_effort_cache",
        per_device_train_batch_size=4,  # 如果 NPU 显存吃紧，可改为 2 或 1
        per_device_eval_batch_size=4,
        report_to="none",
        logging_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    print("⚡ 正在计算目标项的 Loss 作为难度得分...")
    
    # 手动利用 DataLoader 提取每条数据的 Loss 作为 Effort
    effort_scores = []
    dataloader = trainer.get_train_dataloader()
    
    model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            inputs = trainer._prepare_inputs(inputs)
            outputs = model(**inputs)
            
            # 获取 per-sample 维度的 loss 代理 Effort
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size(0), -1)
            
            # 计算每个序列的平均有效 Loss
            valid_tokens = (shift_labels != IGNORE_INDEX).sum(dim=-1)
            seq_loss = loss.sum(dim=-1) / (valid_tokens + 1e-6)
            
            effort_scores.extend(seq_loss.cpu().numpy().tolist())
            
            if step % 50 == 0:
                print(f"  - 进度: {step * training_args.per_device_train_batch_size} / {len(train_dataset)}...")

    # 清理显存
    del model
    del trainer
    torch.cuda.empty_cache()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()

    return effort_scores
