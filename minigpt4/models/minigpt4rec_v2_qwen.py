import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os

from minigpt4.common.registry import registry
from minigpt4.models.rec_model import Rec2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        freeze_proj=False
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num

        print("runing MiniGPT4Rec_v2 ...... ")

        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        # try:
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            print("successfully load the pretrained model......")
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

            

        print('Loading LLama model:', llama_model)
        if "Qwen" in llama_model:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True)
            self.llama_tokenizer.add_special_tokens({"unk_token":""})
            
    def generate_for_samples_v2(self, samples,return_all=False):
        try:
            # sample = samples["image"]
            user_selective_prompts = False
            if hasattr(samples, 'question_split'):  # VQA dataset
                print('VQA Batch')
                raise NotImplementedError("not implement")
                # vqa_prompt = '###Human: <Img><ImageHere></Img> '
                # img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
            elif self.prompt_list:
                if user_selective_prompts:  # automatically setting prompt according to the prompt_flag
                    prompt_flag = samples['prompt_flag']
                    unique_flags = torch.unique(prompt_flag)
                    sample_embeds = []
                    atts_samples = []
                    true_idx = torch.zeros_like(prompt_flag)
                    pre_ = 0
                    for k_flag in unique_flags:
                        idx_k = torch.nonzero(prompt_flag==k_flag)[0]
                        true_idx[idx_k] = pre_ + torch.arange(idx_k.shape[0])
                        pre_ += idx_k.shape[0]
                        sub_k_sample = {}
                        for key_ in samples.keys():
                            sub_k_sample[key_] = samples[key_][idx_k]
                        if k_flag == 0:   # assume the fist prompt does not use ID information, for cold items
                            used_prompt = self.prompt_list[-1]
                        else:
                            used_prompt = self.prompt_list[1] # during inference, use ID+title information by default.
                        sample_embeds_k, atts_samples_k = self.prompt_based_encode_v2(used_prompt, sub_k_sample)
                        sample_embeds.append(sample_embeds_k)
                        atts_samples.append(atts_samples_k)
                        del sub_k_sample, sample_embeds_k, atts_samples_k
                        import gc
                        gc.collect()
                    sample_embeds = torch.cat(sample_embeds, dim=0)
                    atts_samples = torch.cat(atts_samples,dim=0)
                    sample_embeds = sample_embeds[true_idx]
                    atts_samples = atts_samples[true_idx]
            
                    del true_idx, unique_flags, prompt_flag
                    import gc
                    gc.collect()
                else:
                    prompt = self.prompt_list[0]
                    sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt,samples)
                    # id_orders = get_ids_order(prompt)
                    # samples_encode, atts_samples = self.encode_recdata_v2(samples,ids_order=id_orders)
                    # sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)

            self.llama_tokenizer.padding_side = "right"


            device = samples['UserID'].device #samples_encode['User_emb'].device

            pos_ans = self.pos_ans[0]
            neg_ans = self.neg_ans[0]
            ans_ = {1:pos_ans, 0:neg_ans}

            ans_ = {1:pos_ans, 0:neg_ans}

            with torch.no_grad():
                # text = ["### Response: " + ans_[int(t)]  for t in samples["label"]]
                text = [ ans_[int(t)]  for t in samples["label"]]

                to_regress_tokens = self.llama_tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    add_special_tokens=False
                ).to(device)

                t_posi = to_regress_tokens.input_ids.shape[-1] + 1

                # print("labels:",samples["label"],"token:",to_regress_tokens)

                targets = to_regress_tokens.input_ids.masked_fill(
                    to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
                )
                empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)

                # empty_targets = (
                #     torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                #                dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
                # )
                targets = torch.cat([empty_targets, targets], dim=1)
                
                del empty_targets, text
                import gc
                gc.collect()

            if not self.use_lora:
                to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            else:
                to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
            
            del sample_embeds, to_regress_embeds, atts_samples, to_regress_tokens
            import gc
            gc.collect()

            with self.maybe_autocast():
                outputs = self.llama_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
            
            del inputs_embeds, attention_mask, targets
            import gc
            gc.collect()
            
            pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
            logits_ = outputs.logits[:,-t_posi,:][:,pos_ans_id]
            loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())
            
            del outputs
            import gc
            gc.collect()

            if return_all:
                result = (None, logits_.clone())
                del logits_
                return result
            else:
                result = {"loss": loss, 'logits': logits_.clone()}
                del logits_
                return result
                
        except Exception as e:
            print(f"Error in generate_for_samples_v2: {str(e)}")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
