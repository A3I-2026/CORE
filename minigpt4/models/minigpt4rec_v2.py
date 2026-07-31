import logging
import random

import torch
try:
    import torch_npu
except Exception:
    torch_npu = None
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os

from minigpt4.common.registry import registry
from minigpt4.models.rec_model import Rec2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
import re
import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict

def get_ids_order(prompt):
    id_flags = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_>=0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_

def consitence_loss(ori_embs, proj_embs):
    ori_embs = ori_embs.squeeze()
    proj_embs = proj_embs.squeeze()
    ori_similarities = torch.matmul(ori_embs, ori_embs.T)
    # ori_diag = torch.diag(ori_similarities)+1e9
    proj_similarities = torch.matmul(proj_embs, proj_embs.T)
    # proj_diag = torch.diag(proj_similarities)+1e9
    N_ = ori_similarities.shape[0]
    ori_similarities[range(N_), range(N_)] -= 1e9
    proj_similarities[range(N_), range(N_)] -= 1e9
    ori_similarities = torch.softmax(ori_similarities,dim=-1) 
    proj_similarities = torch.softmax(proj_similarities,dim=-1)
    loss = nn.functional.mse_loss(ori_similarities, proj_similarities)
    # loss = -torch.log(proj_similarities+1e-6).mul(ori_similarities).sum(dim=-1).mean() #+ nn.functional.cross_entropy(,)
    # loss = nn.functional.kl_div(proj_similarities, ori_similarities, reduction="batchmean")
    return loss 

class identical_map(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return x*1.0


@registry.register_model("mini_gpt4rec_v2")
class MiniGPT4Rec_v2(Rec2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }

    def __init__(
        self,
        rec_model="MF",
        rec_config=None,
        pretrained_rec=None,
        freeze_rec=True,
        rec_precision='fp16',
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        proj_token_num=1, # the number of tokens that the user/item embedding projected to
        proj_drop=0,
        lora_config=None,
        proj_mid=5,
        freeze_lora=False,
        freeze_proj=True,
        soft_prompt_len=0,
        soft_prompt_bank_size=0,
        soft_prompt_init_path=None,
        use_grad_ckpt=False,
        use_text_prompt=True,
        enable_soft_prompt=True,
        freeze_soft_prompt=False,
        distill_text= "",  
        distill_weight=0.0,
        **kwargs):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num

        print("runing MiniGPT4Rec_v2 ...... ")

        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        
        pretrain_path = None
        if isinstance(pretrained_rec, str) and os.path.exists(pretrained_rec):
            pretrain_path = pretrained_rec
        else:
            try:
                if isinstance(rec_config, dict):
                    candidate = rec_config.get('pretrained_path', None)
                else:
                    candidate = getattr(rec_config, 'pretrained_path', None)
                if isinstance(candidate, str) and os.path.exists(candidate):
                    pretrain_path = candidate
            except Exception:
                pretrain_path = None

        if self.rec_encoder is not None and pretrain_path:
            self.rec_encoder.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
            print(f"successfully load the pretrained rec encoder from {pretrain_path}")
        else:
            print("skip loading pretrained rec encoder (no valid path provided)")
        # except:
        #     # print(pretrained_rec)
        #     # self.rec_encoder.config
        #     raise RuntimeError("Please provide your pretained rec model path or check whether the pretrained model and the defined mode can match each other")
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info("freeze rec encoder")
            print("freeze rec encoder")

        print('Loading Rec_model Done')

            

        print('Loading LLAMA')
        if (torch_npu is not None and hasattr(torch_npu, 'npu') and torch_npu.npu.is_available()) or (hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()):
            self.low_resource = False

        try:
            import os
            from pathlib import Path
            lm_str = str(llama_model).strip()
     
            lm_str = lm_str.rstrip("/")
        
            if os.path.isdir(lm_str):
                lm_resolved = str(Path(lm_str).resolve())
                self.llama_tokenizer = LlamaTokenizer.from_pretrained(lm_resolved, use_fast=False, local_files_only=True)
            else:
                # 作为 Hub 模型 ID 使用（不带前导斜杠）
                self.llama_tokenizer = LlamaTokenizer.from_pretrained(lm_str, use_fast=False)
        except Exception:
            # 回退：直接按原字符串尝试本地加载
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(str(llama_model).rstrip("/"), use_fast=False, local_files_only=True)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = LlamaForCausalLM.from_pretrained(
            str(llama_model).rstrip("/"),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        if use_grad_ckpt:
            try:
                self.llama_model.gradient_checkpointing_enable()
            except Exception:
                pass

        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            print("Setting Lora")
            self.use_lora = True
            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.dropout,
                bias="none",
                task_type="CAUSAL_LM"
            ) 
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
            print("Setting Lora Done")
            if use_grad_ckpt:
                try:
                    self.llama_model_lora.gradient_checkpointing_enable()
                except Exception:
                    pass
        
        if freeze_lora and self.use_lora and hasattr(self, 'llama_model_lora'):
            print("freeze lora...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False

 
        
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            print("type:", type(proj_mid), proj_mid)
            self.llama_proj = nn.Sequential(
                nn.Linear(self.rec_encoder.config.embedding_size, self.rec_encoder.config.embedding_size*int(proj_mid)),  # ml100=>5
                nn.ReLU(),
                # nn.Dropout(proj_drop),
                nn.Linear(self.rec_encoder.config.embedding_size*int(proj_mid), self.llama_model.config.hidden_size * self.proj_token_num),
            )
            # self.llama_proj = nn.Linear(self.rec_encoder.config.embedding_size, self.llama_model.config.hidden_size * self.proj_token_num)
        elif self.rec_encoder is not None and rec_model=="personlized_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("personalized prompt learning....")
            self.llama_proj = nn.Linear(rec_config.item_num+rec_config.user_num, self.llama_model.config.hidden_size * self.proj_token_num,bias=False) #identical_map()
        elif self.rec_encoder is not None and rec_model=="soft_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("soft prompt learning....")
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * self.proj_token_num,bias=False) #identical_map()
        else:
            self.llama_proj = None

       
        
        if freeze_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("!!!! freeze llama_proj...")
        self.disable_rec_input = bool(freeze_proj)

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt=False
    
        self.pos_ans = ['Yes']
        self.neg_ans = ['No']

        self.soft_prompt_len = int(soft_prompt_len) if soft_prompt_len is not None else 0
        self.soft_prompt_bank_size = int(soft_prompt_bank_size) if soft_prompt_bank_size is not None else 0
        self.enable_soft_prompt = bool(enable_soft_prompt)
        self.use_text_prompt = bool(use_text_prompt)
        self.freeze_soft_prompt = bool(freeze_soft_prompt)
        if self.enable_soft_prompt and self.soft_prompt_len > 0:
            self.soft_prompt = nn.Parameter(torch.zeros(self.soft_prompt_len, self.llama_model.config.hidden_size))
            try:
                nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
            except Exception:
                pass
            if self.freeze_soft_prompt:
                self.soft_prompt.requires_grad_(False)
        else:
            self.soft_prompt = None
        if self.enable_soft_prompt and self.soft_prompt_len > 0 and self.soft_prompt_bank_size > 0:
            self.soft_prompt_bank = nn.Parameter(torch.zeros(self.soft_prompt_bank_size, self.soft_prompt_len, self.llama_model.config.hidden_size))
            try:
                nn.init.normal_(self.soft_prompt_bank, mean=0.0, std=0.02)
            except Exception:
                pass
            if self.freeze_soft_prompt:
                self.soft_prompt_bank.requires_grad_(False)
        else:
            self.soft_prompt_bank = None

        if self.enable_soft_prompt and soft_prompt_init_path is not None and os.path.exists(soft_prompt_init_path):
            try:
                obj = torch.load(soft_prompt_init_path, map_location="cpu")
                if isinstance(obj, dict):
                    if "soft_prompt_bank" in obj:
                        arr = obj["soft_prompt_bank"]
                    elif "soft_prompt" in obj:
                        arr = obj["soft_prompt"]
                    else:
                        arr = obj
                elif isinstance(obj, torch.Tensor):
                    arr = obj
                else:
                    arr = torch.tensor(np.load(soft_prompt_init_path))
            except Exception:
                arr = torch.tensor(np.load(soft_prompt_init_path))

            if isinstance(arr, torch.Tensor):
                if arr.ndim == 3:
                    bs, L, H = arr.shape
                    if self.soft_prompt_bank is None or list(self.soft_prompt_bank.shape) != [bs, L, H]:
                        self.soft_prompt_len = int(L)
                        self.soft_prompt_bank_size = int(bs)
                        self.soft_prompt_bank = nn.Parameter(arr.float())
                        self.soft_prompt = None
                        if self.freeze_soft_prompt:
                            self.soft_prompt_bank.requires_grad_(False)
                    else:
                        self.soft_prompt_bank.data.copy_(arr.float())
                elif arr.ndim == 2:
                    L, H = arr.shape
                    if self.soft_prompt is None or list(self.soft_prompt.shape) != [L, H]:
                        self.soft_prompt_len = int(L)
                        self.soft_prompt = nn.Parameter(arr.float())
                        if self.freeze_soft_prompt:
                            self.soft_prompt.requires_grad_(False)
                    else:
                        self.soft_prompt.data.copy_(arr.float())

        self.distill_text = distill_text
        self.distill_weight = float(distill_weight) if distill_weight is not None else 0.0
        if not self.enable_soft_prompt:
            self.soft_prompt_len = 0
            self.soft_prompt_bank_size = 0
            self.soft_prompt = None
            self.soft_prompt_bank = None
            self.distill_weight = 0.0

     
        if prompt_path:
            try:
                import os
                from pathlib import Path
                repo = Path(__file__).resolve().parents[2]
                base = repo/ 'prompts'
                candidates = [str(prompt_path), str(base/'softprompt_amazon.txt'), str(base/'core_amazon.txt'), str(base/'softprompt.txt'), str(base/'core_movie.txt')]
                chosen = None
                for p in candidates:
                    try:
                        if os.path.isfile(str(p)):
                            chosen = str(p)
                            break
                    except Exception:
                        pass
                if chosen is not None:
                    with open(chosen, 'r', encoding='utf-8') as f:
                        raw_prompts = f.read().splitlines()
                    filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
                    self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
                    print('Load {} training prompts'.format(len(self.prompt_list)))
                    print('Prompt List: \n{}'.format(self.prompt_list))
                    self.has_pri_decode=False
                    self.prompt_list_p = None
                else:
                    self.prompt_list = []
                    self.prompt_list_p = None
            except Exception:
                self.prompt_list = []
                self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None

        self.distill_weight = distill_weight
        self.distill_text = distill_text
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt_bank_size = soft_prompt_bank_size
        self.use_text_prompt = use_text_prompt


    def to_be_trained(self):
        if self.use_lora:
            return True
        # return True # have lora module, will be trained anyway
        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True
        ### No ID is used, disable the projection layers
        # self.llama_proj = None
        # for name, param in self.llama_proj.named_parameters():
        #     param.requires_grad = False  
        return False
    
    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def set_answer_type(self,mode):
        if mode is None:
            return
        if mode == 'v1':
        # pos_ans = ["The former item.", "The first item.", "The former.", "The first.", "The former one.", "The first one."]
        # neg_ans = ["The latter item.", "The second item.", "The latter.", "The second.", "The latter one.", "The second one."]
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            # self.pos_ans = ['enjoy']
            # self.neg_ans = ['dislike']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[-1]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[-1]
            print("answer token ids: pos:",pos_ans_id, "neg ids:", neg_ans_id)
            
        else:
       
            return
    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list),self.pos_ans[0],self.neg_ans[0]))


    def encode_recdata_v1(self, sample): # used for stage1
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            all_user_embeds, all_items_embeds = self.rec_encoder.computer()
            user_embeds = self.rec_encoder.user_encoder(sample['UserID'],all_users=all_user_embeds).unsqueeze(-2)

            tgt_key = 'PairItemIDs' if 'PairItemIDs' in sample else 'TargetItemID'
            targetItem_embed = self.rec_encoder.item_encoder(sample[tgt_key], all_items=all_items_embeds)
            

            user_embeds_llama = self.llama_proj(user_embeds)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed)
        
        sample_embeds_llama = {
            'User_emb': user_embeds_llama,
            'PairItem_emb': targetItem_embeds_llama,
        }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def encode_recdata_v2(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            try:
                rec_dev = next(self.rec_encoder.parameters()).device
            except Exception:
                rec_dev = device
            try:
                self.rec_encoder = self.rec_encoder.to(rec_dev)
            except Exception:
                pass
            if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                """
                not really user embeding, but the embedding merged for one sample point
                """
                user_embeds = self.rec_encoder.all_encode(sample['UserID'],sample['TargetItemID'],sample['sas_seq'][:,-10:]).unsqueeze(-2)
            else:
                uids = sample['UserID']
                try:
                    uids = uids.to(rec_dev)
                    if uids.dtype not in (torch.int64, torch.int32):
                        uids = uids.long()
                except Exception:
                    uids = uids.long()
                try:
                    if hasattr(self.rec_encoder, 'config') and hasattr(self.rec_encoder.config, 'user_num'):
                        uids = uids.clamp(min=0, max=int(self.rec_encoder.config.user_num)-1)
                except Exception:
                    pass
                user_embeds = self.rec_encoder.user_encoder(uids, all_users=all_user_embeds).unsqueeze(-2)
            # ***Note: here, for sasrec, item embedding comes form the last layer 
            tids = sample['TargetItemID']
            try:
                tids = tids.to(rec_dev)
                if tids.dtype not in (torch.int64, torch.int32):
                    tids = tids.long()
            except Exception:
                tids = tids.long()
            try:
                if hasattr(self.rec_encoder, 'config') and hasattr(self.rec_encoder.config, 'item_num'):
                    tids = tids.clamp(min=0, max=int(self.rec_encoder.config.item_num)-1)
            except Exception:
                pass
            targetItem_embed = self.rec_encoder.item_encoder(tids, all_items=all_item_embeds).unsqueeze(-2)
            
            

            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            # if self.rec_encoder !="DCN":
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size,-1, self.proj_token_num, hidden_size)
            
            # loss_c = consitence_loss(user_embeds, user_embeds_llama) + consitence_loss(targetItem_embed, targetItem_embeds_llama)
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order)==3:
                hist = sample['InteractedItemIDs_pad']
                try:
                    hist = hist.to(rec_dev)
                    if hist.dtype not in (torch.int64, torch.int32):
                        hist = hist.long()
                except Exception:
                    hist = hist.long()
                try:
                    if hasattr(self.rec_encoder, 'config') and hasattr(self.rec_encoder.config, 'item_num'):
                        hist = hist.clamp(min=0, max=int(self.rec_encoder.config.item_num)-1)
                except Exception:
                    pass
                interactedItem_embeds = self.rec_encoder.item_encoder(hist, all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size,-1, self.proj_token_num, hidden_size)

                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds,dim=1)              
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad']==self.rec_encoder.padding_index, 0, idx_flag) # indx_of_paddded historical items
                # to indicate user_id, his_items_id, target_item_id
                idx_flag = [torch.ones([idx_flag.shape[0],1]).to(idx_flag.device),idx_flag,torch.ones([idx_flag.shape[0],1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag,dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)



                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:,0],idx_nopad[:,1]].reshape(-1, hidden_size),
                    # 'loss_c': loss_c
                }
            else:
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                    # 'loss_c': loss_c
                }
        sample_atts_llama = None
        # {
        #     'user': atts_user,
        #     'TargetItem': atts_targetItem,
        #     'InteractedItems': atts_interactedItem
        # }
        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v1(self, samples, ori_samples, atts_sample, prompt):
        if self.disable_rec_input:
            bsz = ori_samples['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            empty = torch.zeros((bsz, 0, hidden_size), dtype=torch.float16, device=ori_samples['UserID'].device)
            atts = torch.ones((bsz, 0), dtype=torch.long, device=ori_samples['UserID'].device)
            return empty, atts
        return samples, atts_sample

    def prompt_based_encode_v2(self, prompt, samples):
        if self.disable_rec_input:
            bsz = samples['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            sample_embeds = torch.zeros((bsz, 0, hidden_size), dtype=torch.float16, device=samples['UserID'].device)
            atts = torch.ones((bsz, 0), dtype=torch.long, device=samples['UserID'].device)
            return sample_embeds, atts
        ids_order = get_ids_order(prompt)
        samples_encode, atts_samples = self.encode_recdata_v2(samples, ids_order=ids_order)
        sample_embeds = torch.cat([
            samples_encode['User_emb'],
            samples_encode['TargetItem_emb']
        ], dim=1)
        atts = torch.ones(sample_embeds.size()[:-1], dtype=torch.long, device=sample_embeds.device)
        return sample_embeds, atts

    def forward(self, samples):
        mode = getattr(self, 'run_mode_', None)
        if mode == 'v1':
            return self.forward_v1(samples)
        return self.forward_v2(samples)

    def forward_v1(self, samples):
        if self.disable_rec_input:
            bsz = samples['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            sample_embeds = torch.zeros((bsz, 0, hidden_size), dtype=torch.float16, device=samples['UserID'].device)
            atts_samples = torch.ones((bsz, 0), dtype=torch.long, device=samples['UserID'].device)
        else:
            samples_encode, atts_samples = self.encode_recdata_v1(samples)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, None)
        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        try:
            if sample_embeds.device != device:
                sample_embeds = sample_embeds.to(device)
            if atts_samples.device != device:
                atts_samples = atts_samples.to(device)
        except Exception:
            pass
        ans_ = {1: self.pos_ans, 0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples['label']]
        tokens = self.llama_tokenizer(text, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_txt_len, add_special_tokens=False).to(device)
        targets = tokens.input_ids.masked_fill(tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.int32, device=device)
        empty_targets.fill_(-100)
        empty_targets = empty_targets.to(torch.long)
        if self.soft_prompt is not None or self.soft_prompt_bank is not None:
            soft_len = self.soft_prompt_len if self.soft_prompt_len > 0 else 0
            soft_empty = torch.ones([atts_samples.shape[0], soft_len], dtype=torch.int32, device=device)
            soft_empty.fill_(-100)
            soft_empty = soft_empty.to(torch.long)
            targets = torch.cat([soft_empty, empty_targets, targets], dim=1)
        else:
            targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self._safe_embed_tokens(tokens.input_ids, device)
        use_bank = self.soft_prompt_bank is not None and ('prompt_flag' in samples or 'UserID' in samples)
        if self.soft_prompt is not None or use_bank:
            if use_bank:
                if 'prompt_flag' in samples:
                    prompt_ids = torch.nan_to_num(samples.get('prompt_flag'), nan=0.0).long().clamp(min=0, max=self.soft_prompt_bank_size-1)
                else:
                    prompt_ids = (torch.nan_to_num(samples.get('UserID'), nan=0.0).long() % self.soft_prompt_bank_size).clamp(min=0, max=self.soft_prompt_bank_size-1)
                try:
                    prompt_ids = prompt_ids.to(device)
                except Exception:
                    pass
                bank = self.soft_prompt_bank
                try:
                    bank = bank.to(device)
                except Exception:
                    pass
                soft = bank[prompt_ids]
            else:
                soft = self.soft_prompt
                try:
                    soft = soft.to(device)
                except Exception:
                    pass
                soft = soft.unsqueeze(0).expand(sample_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([soft, sample_embeds, to_regress_embeds], dim=1)
            soft_atts = torch.ones([atts_samples.shape[0], self.soft_prompt_len], dtype=torch.int32, device=device).to(torch.long)
            attention_mask = torch.cat([soft_atts, atts_samples, tokens.attention_mask], dim=1)
        else:
            inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_samples, tokens.attention_mask], dim=1)
        try:
            inputs_embeds = inputs_embeds.clone()
            attention_mask = attention_mask.clone()
        except Exception:
            pass
        with self.maybe_autocast():
            outputs = (self.llama_model_lora if self.use_lora else self.llama_model)(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=targets)
        loss = outputs.loss
        if self.distill_weight > 0 and (self.soft_prompt is not None or use_bank) and self.soft_prompt_len > 0 and isinstance(self.distill_text, str) and len(self.distill_text)>0:
            teacher_tok = self.llama_tokenizer(self.distill_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.soft_prompt_len, add_special_tokens=False).to(device)
            teacher_emb = self._safe_embed_tokens(teacher_tok.input_ids, device)
           
            bsz = sample_embeds.size(0)
            if teacher_emb.shape[0] != bsz:
                teacher_emb = teacher_emb.repeat(bsz, 1, 1)
            if use_bank:
                soft_current = soft
            elif self.soft_prompt is not None:
                soft_current = self.soft_prompt.unsqueeze(0).expand(sample_embeds.size(0), -1, -1)
            else:
                soft_current = None
            if soft_current is not None:
                if teacher_emb.shape[1] != soft_current.shape[1]:
                    L = min(teacher_emb.shape[1], soft_current.shape[1])
                    teacher_emb = teacher_emb[:, :L, :]
                    soft_current = soft_current[:, :L, :]
            
                sc = torch.nan_to_num(soft_current.to(torch.float32), nan=0.0, posinf=1e3, neginf=-1e3)
                te = torch.nan_to_num(teacher_emb.to(torch.float32), nan=0.0, posinf=1e3, neginf=-1e3)
                loss = loss + self.distill_weight * nn.functional.mse_loss(sc, te)
        return {"loss": loss}

    def set_stage2_training(self):

        print(">>> Switching to Stage 2: Joint Adaptation (Distillation + CIE Tuning) ...")
        
      
        for name, param in self.named_parameters():
            param.requires_grad = False
            
   
        if self.soft_prompt is not None:
            self.soft_prompt.requires_grad = True
            print(f" -> Unfrozen: Soft Prompt ({self.soft_prompt.shape})")
      
        if hasattr(self, 'llama_proj'):
            for param in self.llama_proj.parameters():
                param.requires_grad = True
            print(" -> Unfrozen: CIE Projection Layer (llama_proj)")
        else:
            print(" [WARNING] 'llama_proj' not found! Please check variable name in Rec2Base.")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" -> Total Trainable Params: {trainable_params}")


    def forward_v2(self, samples):
       
        temp_prompt = self.prompt_list[0] if len(self.prompt_list) > 0 else ""
        sample_embeds, atts_samples = self.prompt_based_encode_v2(temp_prompt, samples)

        device = sample_embeds.device
       
        distill_loss = torch.tensor(0.0, device=device)
        
        if self.training and self.distill_weight > 0 and self.soft_prompt is not None:
            if not self.distill_text:
                self.distill_text = "Recommend an item for the user based on the history."
            
            with torch.no_grad(): 
                teacher_tokens = self.llama_tokenizer(
                    self.distill_text, 
                    return_tensors='pt', 
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.soft_prompt_len, 
                    add_special_tokens=False
                ).to(device)
                
                teacher_embeds = self._safe_embed_tokens(teacher_tokens.input_ids, device)
                
                bsz = sample_embeds.shape[0]
                if teacher_embeds.shape[0] != bsz:
                    teacher_embeds = teacher_embeds.expand(bsz, -1, -1)

        inputs_embeds_list = []
        attention_mask_list = []
        
        # Soft Prompt (Student)
        if self.soft_prompt is not None:
            if self.soft_prompt_bank is not None and ('prompt_flag' in samples or 'UserID' in samples):
                if 'prompt_flag' in samples:
                    prompt_ids = torch.nan_to_num(samples.get('prompt_flag'), nan=0.0).long().clamp(min=0, max=self.soft_prompt_bank_size-1)
                else:
                    prompt_ids = (torch.nan_to_num(samples.get('UserID'), nan=0.0).long() % self.soft_prompt_bank_size).clamp(min=0, max=self.soft_prompt_bank_size-1)
                soft_student = self.soft_prompt_bank[prompt_ids.to(device)]
            else:
                soft_student = self.soft_prompt.unsqueeze(0).expand(sample_embeds.shape[0], -1, -1)
            
            curr_len = soft_student.shape[1]
            tgt_len = int(self.soft_prompt_len)
            soft_student = soft_student[:, :min(curr_len, tgt_len), :]
            
            inputs_embeds_list.append(soft_student)
            attention_mask_list.append(torch.ones(soft_student.shape[:2], device=device).long())
            
          
            if self.training and self.distill_weight > 0:
                min_len = min(soft_student.shape[1], teacher_embeds.shape[1])
                student_part = soft_student[:, :min_len, :].to(torch.float32)
                teacher_part = teacher_embeds[:, :min_len, :].to(torch.float32)
                student_norm = torch.nn.functional.normalize(student_part, p=2, dim=-1)
                teacher_norm = torch.nn.functional.normalize(teacher_part, p=2, dim=-1)
                distill_loss = self.distill_weight * torch.nn.functional.mse_loss(
                    student_norm,
                    teacher_norm
                )

        inputs_embeds_list.append(sample_embeds)
        attention_mask_list.append(atts_samples)

    
        start_tok = self.llama_tokenizer(" ", add_special_tokens=False).input_ids[0]
        start_tokens = torch.full((sample_embeds.shape[0], 1), start_tok, device=device).long()
        start_embeds = self._safe_embed_tokens(start_tokens, device)
        
        inputs_embeds_list.append(start_embeds)
        attention_mask_list.append(torch.ones_like(start_tokens))

    
        final_inputs = torch.cat(inputs_embeds_list, dim=1)
        final_atts = torch.cat(attention_mask_list, dim=1)

     
        model_to_run = self.llama_model_lora if self.use_lora else self.llama_model
        
        
        
        outputs = model_to_run(
            inputs_embeds=final_inputs, 
            attention_mask=final_atts, 
            return_dict=True
        )
        
        bsz = sample_embeds.shape[0]
        try:
            am = final_atts[:, :outputs.logits.shape[1]]
            valid_last = am.sum(dim=1).clamp(min=1) - 1
        except Exception:
            valid_last = torch.full((outputs.logits.shape[0],), max(0, outputs.logits.shape[1]-1), dtype=torch.long, device=outputs.logits.device)
        idx = valid_last.to(outputs.logits.device)
        logits = outputs.logits[torch.arange(bsz, device=idx.device), idx, :]

        pos_str = getattr(self, "pos_ans", ["Yes"])[0]
        neg_str = getattr(self, "neg_ans", ["No"])[0]
        pos_id = self.llama_tokenizer(" " + pos_str, add_special_tokens=False).input_ids[-1]
        neg_id = self.llama_tokenizer(" " + neg_str, add_special_tokens=False).input_ids[-1]

        logits_pair = logits[:, [neg_id, pos_id]]
        logits_pair = torch.nan_to_num(logits_pair, nan=0.0, posinf=1.0, neginf=0.0).float()

        if 'label' in samples:
            labels = torch.nan_to_num(samples['label'], nan=0.0).to(device).long()
        elif 'targets' in samples:
            targets_text = samples['targets']
            pos_str_l = str(pos_str).lower()
            labels_list = []
            for t in targets_text:
                labels_list.append(1 if pos_str_l in str(t).lower() else 0)
            labels = torch.tensor(labels_list, device=device).long()
        else:
            print("[Warning] No 'label' or 'targets' found in samples! Defaulting to zeros.")
            labels = torch.zeros(bsz, dtype=torch.long, device=device)

        rec_loss = torch.nn.functional.cross_entropy(logits_pair.to(torch.float32), labels)

        probs = torch.softmax(logits_pair.to(torch.float32), dim=-1)[:, 1]

        total_loss = rec_loss + distill_loss
        probs = torch.softmax(logits_pair, dim=-1)[:, 1]
        return {"loss": total_loss, "rec_loss": rec_loss.detach(), "distill_loss": distill_loss.detach(), "logits": probs.detach(),  # 返回预测概率
            "labels": labels.detach()  
             }



    def generate_for_samples(self, samples):
        mode = getattr(self, 'run_mode_', None)
        if mode == 'v1':
            return self.generate_for_samples_v1(samples)
        return self.generate_for_samples_v2(samples)

    def generate_for_samples_v1(self, samples):
        if self.disable_rec_input:
            bsz = samples['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            sample_embeds = torch.zeros((bsz, 0, hidden_size), dtype=torch.float16, device=samples['UserID'].device)
            atts_samples = torch.ones((bsz, 0), dtype=torch.long, device=samples['UserID'].device)
        else:
            prompt = self.prompt_list[0] if len(self.prompt_list) > 0 else ""
            sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt, samples)
        self.llama_tokenizer.padding_side = "right"
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                dev_id = torch.npu.current_device() if hasattr(torch.npu, 'current_device') else 0
                device = torch.device(f'npu:{dev_id}')
            else:
                device = samples['UserID'].device
        except Exception:
            device = samples['UserID'].device
        try:
            if next(self.llama_model.parameters()).device != device:
                self.llama_model = self.llama_model.to(device)
                self.llama_model.model = self.llama_model.model.to(device)
                self.llama_model.model.embed_tokens = self.llama_model.model.embed_tokens.to(device)
        except Exception:
            pass
        start_tok = self.llama_tokenizer(" ", add_special_tokens=False).input_ids[0]
        start_tokens = torch.full((samples['UserID'].shape[0], 1), start_tok, dtype=torch.int32, device=device).to(torch.long)
        start_embeds = self._safe_embed_tokens(start_tokens, device)
        use_bank = self.soft_prompt_bank is not None and ('prompt_flag' in samples or 'UserID' in samples)
        if self.soft_prompt is not None or use_bank:
            if use_bank:
                if 'prompt_flag' in samples:
                    prompt_ids = torch.nan_to_num(samples.get('prompt_flag'), nan=0.0).long().clamp(min=0, max=self.soft_prompt_bank_size-1)
                else:
                    prompt_ids = (torch.nan_to_num(samples.get('UserID'), nan=0.0).long() % self.soft_prompt_bank_size).clamp(min=0, max=self.soft_prompt_bank_size-1)
                soft = self.soft_prompt_bank[prompt_ids]
            else:
                soft = self.soft_prompt.unsqueeze(0).expand(sample_embeds.size(0), -1, -1)
            try:
                target_len = int(self.soft_prompt_len) if isinstance(self.soft_prompt_len, int) else soft.shape[1]
            except Exception:
                target_len = soft.shape[1]
            soft = soft[:, :min(target_len, soft.shape[1]), :]
            inputs_embeds = torch.cat([soft, sample_embeds, start_embeds], dim=1)
            soft_atts = torch.ones([atts_samples.shape[0], self.soft_prompt_len], dtype=torch.int32, device=device).to(torch.long)
            attention_mask = torch.cat([soft_atts, atts_samples, torch.ones_like(start_tokens, dtype=torch.int32, device=device).to(torch.long)], dim=1)
            sample_len = atts_samples.shape[1] + self.soft_prompt_len
        else:
            inputs_embeds = torch.cat([sample_embeds, start_embeds], dim=1)
            attention_mask = torch.cat([atts_samples, torch.ones_like(start_tokens, dtype=torch.int32, device=device).to(torch.long)], dim=1)
            sample_len = atts_samples.shape[1]
        with self.maybe_autocast():
            outputs = (self.llama_model_lora if self.use_lora else self.llama_model)(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        try:
            if getattr(self, '_pos_id', None) is None or getattr(self, '_neg_id', None) is None:
                self._pos_id = self.llama_tokenizer(" " + self.pos_ans[0], add_special_tokens=False).input_ids[-1]
                self._neg_id = self.llama_tokenizer(" " + self.neg_ans[0], add_special_tokens=False).input_ids[-1]
            pos_id = self._pos_id
            neg_id = self._neg_id
        except Exception:
            pos_id = self.llama_tokenizer(" " + self.pos_ans[0], add_special_tokens=False).input_ids[-1]
            neg_id = self.llama_tokenizer(" " + self.neg_ans[0], add_special_tokens=False).input_ids[-1]
        vocab = outputs.logits.shape[-1]
        pos_id = int(pos_id) % vocab
        neg_id = int(neg_id) % vocab
        if pos_id == neg_id:
            neg_id = (neg_id + 1) % vocab
        try:
            am = attention_mask[:, :outputs.logits.shape[1]]
            valid_last = am.sum(dim=1).clamp(min=1) - 1
        except Exception:
            valid_last = torch.full((outputs.logits.shape[0],), max(0, outputs.logits.shape[1]-1), dtype=torch.long, device=outputs.logits.device)
        idx = valid_last.to(outputs.logits.device)
        bsz = outputs.logits.shape[0]
        logits_pair = outputs.logits[torch.arange(bsz, device=idx.device), idx, :][:, [neg_id, pos_id]]
        logits_pair = torch.nan_to_num(logits_pair, nan=0.0, posinf=1.0, neginf=0.0).float()
        labels_ce = torch.nan_to_num(samples['label'], nan=0.0).long().clamp(min=0, max=1)
        loss = nn.functional.cross_entropy(logits_pair, labels_ce)
        probs = torch.softmax(logits_pair, dim=-1)[:, 1]
        return {"loss": loss, "logits": probs}

    def generate_for_samples_v2(self, samples):
        prompt = self.prompt_list[0] if len(self.prompt_list) > 0 else ""
        sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt, samples)
        self.llama_tokenizer.padding_side = "right"
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                dev_id = torch.npu.current_device() if hasattr(torch.npu, 'current_device') else 0
                device = torch.device(f'npu:{dev_id}')
            else:
                device = samples['UserID'].device
        except Exception:
            device = samples['UserID'].device
        try:
            if next(self.llama_model.parameters()).device != device:
                self.llama_model = self.llama_model.to(device)
                self.llama_model.model = self.llama_model.model.to(device)
                self.llama_model.model.embed_tokens = self.llama_model.model.embed_tokens.to(device)
        except Exception:
            pass
        if True:
            inputs_embeds_list = []
            if self.soft_prompt is not None:
                if self.soft_prompt_bank is not None and ('prompt_flag' in samples or 'UserID' in samples):
                    if 'prompt_flag' in samples:
                        prompt_ids = torch.nan_to_num(samples.get('prompt_flag'), nan=0.0).long().clamp(min=0, max=self.soft_prompt_bank_size-1)
                    else:
                        prompt_ids = (torch.nan_to_num(samples.get('UserID'), nan=0.0).long() % self.soft_prompt_bank_size).clamp(min=0, max=self.soft_prompt_bank_size-1)
                    soft_student = self.soft_prompt_bank[prompt_ids.to(device)]
                else:
                    soft_student = self.soft_prompt.unsqueeze(0).expand(sample_embeds.shape[0], -1, -1)
                soft_student = soft_student[:, :int(self.soft_prompt_len), :]
                inputs_embeds_list.append(soft_student)
            inputs_embeds_list.append(sample_embeds)
            start_tok = self.llama_tokenizer(" ", add_special_tokens=False).input_ids[0]
            start_tokens = torch.full((sample_embeds.shape[0], 1), start_tok, device=device, dtype=torch.long)
            start_embeds = self._safe_embed_tokens(start_tokens, device)
            inputs_embeds_list.append(start_embeds)
            inputs_embeds = torch.cat(inputs_embeds_list, dim=1)

            with self.maybe_autocast():
                model_to_run = self.llama_model_lora if self.use_lora else self.llama_model
                outputs = model_to_run.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            pos_id = self.llama_tokenizer(" " + self.pos_ans[0], add_special_tokens=False).input_ids[-1]
            neg_id = self.llama_tokenizer(" " + self.neg_ans[0], add_special_tokens=False).input_ids[-1]
            first_token_scores = outputs.scores[0]
            logits_pair = first_token_scores[:, [neg_id, pos_id]]
            probs = torch.softmax(logits_pair, dim=-1)[:, 1]
            return {"loss": torch.tensor(0.0, device=device), "logits": probs}
        else:
            try:
                tmpl = prompt or ""
                item_titles = samples.get('InteractedItemTitles', samples.get('PairItemTitles', ''))
                tgt_title = samples.get('TargetItemTitle', '')
                if isinstance(item_titles, torch.Tensor):
                    item_titles = ""
                if isinstance(tgt_title, torch.Tensor):
                    tgt_title = ""
                q_text = tmpl.replace('<ItemTitleList>', str(item_titles)).replace('<TargetItemTitle>', str(tgt_title))
                q_tokens = self.llama_tokenizer(q_text, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_txt_len, add_special_tokens=False).to(device)
                q_embeds = self._safe_embed_tokens(q_tokens.input_ids, device)
                q_atts = q_tokens.attention_mask
                bsz = samples['UserID'].shape[0]
                if q_embeds.shape[0] != bsz:
                    q_embeds = q_embeds.expand(bsz, -1, -1).contiguous()
                    q_atts = q_atts.expand(bsz, -1).contiguous()
            except Exception:
                q_embeds = torch.zeros((samples['UserID'].shape[0], 0, self.llama_model.config.hidden_size), dtype=torch.float16, device=device)
                q_atts = torch.ones((samples['UserID'].shape[0], 0), dtype=torch.long, device=device)
        start_tok = self.llama_tokenizer(" ", add_special_tokens=False).input_ids[0]
        start_tokens = torch.full((samples['UserID'].shape[0], 1), start_tok, dtype=torch.long, device=device)
        start_embeds = self._safe_embed_tokens(start_tokens, device)
        use_bank = self.soft_prompt_bank is not None and ('prompt_flag' in samples or 'UserID' in samples)
        if self.soft_prompt is not None or use_bank:
            if use_bank:
                if 'prompt_flag' in samples:
                    prompt_ids = torch.nan_to_num(samples.get('prompt_flag'), nan=0.0).long().clamp(min=0, max=self.soft_prompt_bank_size-1)
                else:
                    prompt_ids = (torch.nan_to_num(samples.get('UserID'), nan=0.0).long() % self.soft_prompt_bank_size).clamp(min=0, max=self.soft_prompt_bank_size-1)
                soft = self.soft_prompt_bank[prompt_ids]
            else:
                soft = self.soft_prompt.unsqueeze(0).expand(sample_embeds.size(0), -1, -1)
            try:
                target_len = int(self.soft_prompt_len) if isinstance(self.soft_prompt_len, int) else soft.shape[1]
            except Exception:
                target_len = soft.shape[1]
            soft = soft[:, :min(target_len, soft.shape[1]), :]
            inputs_embeds = torch.cat([soft, sample_embeds, q_embeds, start_embeds], dim=1)
            soft_atts = torch.ones([atts_samples.shape[0], self.soft_prompt_len], dtype=torch.int32, device=device).to(torch.long)
            attention_mask = torch.cat([soft_atts, atts_samples, q_atts, torch.ones_like(start_tokens)], dim=1)
            sample_len = atts_samples.shape[1] + q_atts.shape[1] + self.soft_prompt_len
        else:
            inputs_embeds = torch.cat([sample_embeds, q_embeds, start_embeds], dim=1)
            attention_mask = torch.cat([atts_samples, q_atts, torch.ones_like(start_tokens)], dim=1)
            sample_len = atts_samples.shape[1] + q_atts.shape[1]
        with self.maybe_autocast():
            outputs = (self.llama_model_lora if self.use_lora else self.llama_model)(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        pos_id = self.llama_tokenizer(" " + self.pos_ans[0], add_special_tokens=False).input_ids[-1]
        neg_id = self.llama_tokenizer(" " + self.neg_ans[0], add_special_tokens=False).input_ids[-1]
        vocab = outputs.logits.shape[-1]
        pos_id = int(pos_id) % vocab
        neg_id = int(neg_id) % vocab
        if pos_id == neg_id:
            neg_id = (neg_id + 1) % vocab
        if sample_len >= outputs.logits.shape[1]:
            sample_len = max(0, outputs.logits.shape[1] - 1)
        logits_pair = outputs.logits[:, sample_len, :][:, [neg_id, pos_id]]
        logits_pair = torch.nan_to_num(logits_pair, nan=0.0, posinf=1.0, neginf=0.0).float()
        labels_ce = torch.nan_to_num(samples['label'], nan=0.0).long().clamp(min=0, max=1)
        loss = nn.functional.cross_entropy(logits_pair, labels_ce)
        probs = torch.softmax(logits_pair, dim=-1)[:, 1]
        return {"loss": loss, "logits": probs}
    def _safe_embed_tokens(self, input_ids, device):
        try:
            return self.llama_model.model.embed_tokens(input_ids)
        except Exception:
            ids_cpu = input_ids.to('cpu')
            emb_cpu = self.llama_model.model.embed_tokens(ids_cpu)
            return emb_cpu.to(device)

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        user_num = cfg.get("user_num")
        item_num = cfg.get("item_num")
        ans_type = cfg.get("ans_type")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        lora_config = cfg.get("lora_config", None)
        rec_config = cfg.get("rec_config", None)
        
        enable_soft_prompt = cfg.get("enable_soft_prompt", False)
        soft_prompt_len = cfg.get("soft_prompt_len", 0)
        soft_prompt_init_path = cfg.get("soft_prompt_init_path", "")
        use_text_prompt = cfg.get("use_text_prompt", True)
        soft_prompt_bank_size = cfg.get("soft_prompt_bank_size", 0)

        distill_weight = cfg.get("distill_weight", 0.0)
        distill_text = cfg.get("distill_text", "")
        freeze_proj = cfg.get("freeze_proj", True)

        model = cls(
            llama_model=llama_model,
            user_num=user_num,
            item_num=item_num,
            ans_type=ans_type,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            lora_config=lora_config,
            rec_config=rec_config,
            enable_soft_prompt=enable_soft_prompt,
            soft_prompt_len=soft_prompt_len,
            soft_prompt_init_path=soft_prompt_init_path,
            use_text_prompt=use_text_prompt,
            soft_prompt_bank_size=soft_prompt_bank_size,
            
            distill_weight=distill_weight,
            distill_text=distill_text,
            freeze_proj=freeze_proj
        )

        return model
