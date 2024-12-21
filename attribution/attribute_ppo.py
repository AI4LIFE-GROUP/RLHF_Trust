import torch
import torch.nn as nn
import numpy as np
import os, pickle
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
import json
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from safetensors.torch import load_file
from collections import defaultdict
from time import time
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
LORA_OUT_DIR = ...

class RewardModel(nn.Module):
    def __init__(self, model, v_head_weights, eos_token_id):
        super().__init__()
        self.model = model
        self.transformer = self.model.transformer
        self.v_head = nn.Linear(self.model.config.n_embd, 1, bias=False)
        self.v_head.weight.data = v_head_weights
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        states = self.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        returns = rewards[:, -1]
        return returns

def load_pickle_safely(filename):
    data = []
    with open(filename, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
            except pickle.UnpicklingError:
                print("Unpickling error encountered.")
                break
    return data

def normalize_infl(arr):
    # Copy the array to avoid modifying the original
    normalized_arr = np.zeros_like(arr, dtype=np.float64)

    # Positive values normalization (0, 1)
    pos_mask = arr > 0
    if np.any(pos_mask):
        pos_vals = arr[pos_mask]
        normalized_arr[pos_mask] = pos_vals / np.max(pos_vals)  # Scale by the maximum positive value

    # Negative values normalization (-1, 0)
    neg_mask = arr < 0
    if np.any(neg_mask):
        neg_vals = arr[neg_mask]
        normalized_arr[neg_mask] = neg_vals / -np.min(neg_vals)  # Scale by the minimum (most negative) value

    return normalized_arr
        
rm_name = ...
rm_cache_dir = ...

config = PeftConfig.from_pretrained(os.path.abspath(LORA_OUT_DIR))
lora_base = AutoModelForCausalLM.from_pretrained(rm_name, cache_dir=rm_cache_dir)
lora_base = PeftModel.from_pretrained(lora_base, os.path.abspath(LORA_OUT_DIR))

rm_tokenizer = AutoTokenizer.from_pretrained(rm_name, cache_dir=rm_cache_dir)
rm_tokenizer.pad_token = rm_tokenizer.eos_token

rm_state_dict = load_file(...)
v_head_weights = rm_state_dict['v_head']

rm = RewardModel(lora_base, v_head_weights, rm_tokenizer.eos_token_id)
rm = rm.to(device)

rm_dataset_raw = load_dataset(...)

trust_data_files = [
    ...
]
trust_data_files_old = [
    ...
]
trust_dataset = []
trust_data_num = len(trust_data_files)
for j in range(len(trust_data_files)):
    df = pd.read_csv(trust_data_files[j], lineterminator='\n')
    df_old = pd.read_csv(trust_data_files_old[j], lineterminator='\n')
    for i in tqdm(range(0, len(df), 3)):
        prompt = df.iloc[i]["prompt"]
        continuation = df.iloc[i]["continuation"]
        text = f"Human: {prompt}\nAssistant: {continuation}"
        tokenized_text = rm_tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids

        continuation_old = df_old.iloc[i]["continuation"]
        text_old = f"Human: {prompt}\nAssistant: {continuation_old}"
        tokenized_text_old = rm_tokenizer(text_old, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids

        trust_dataset.append((tokenized_text, tokenized_text_old))
        if len(trust_dataset) >= trust_data_num:
            break
    if len(trust_dataset) >= trust_data_num:
        break
print("Metric dataset size: ", len(trust_dataset))

rm_dataset = []
rm_data_num = len(rm_dataset_raw)
for i in tqdm(range(rm_data_num)):
    chosen_text = rm_dataset_raw[i]['prompt'] + rm_dataset_raw[i]["chosen"]
    rejected_text = rm_dataset_raw[i]['prompt'] + rm_dataset_raw[i]["rejected"]
    tokenized_chosen = rm_tokenizer(chosen_text, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids
    tokenized_rejected = rm_tokenizer(rejected_text, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids
    rm_dataset.append((tokenized_chosen, tokenized_rejected))
print("RM dataset size: ", len(rm_dataset))

names = []
for name, param in rm.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        names.append(name)
        param.requires_grad = True

# print(len(names), names)
selected_modules = names[288:-2] # decide which layers to include in the attribution
print(len(selected_modules), selected_modules)
rm.train()

tr_grad_dict = {}
tr_A_zero = 0
tr_A = 0
tr_B_zero = 0
tr_B = 0

grad_path_rm = ...
save_interval = 10000  # Number of iterations after which to save

# Check if a previous checkpoint exists
if os.path.exists(grad_path_rm):
    with open(grad_path_rm, 'rb') as f:
        grads_rm = pickle.load(f)
    start_idx = grads_rm['start_idx']
    tr_grad_dict = grads_rm['tr_grad_dict']
    tr_A, tr_A_zero, tr_B, tr_B_zero = grads_rm['counts']
else:
    start_idx = 0
    tr_grad_dict = {}
    tr_A, tr_A_zero, tr_B, tr_B_zero = 0, 0, 0, 0

for i in tqdm(range(start_idx, len(rm_dataset))):
    if start_idx >= len(rm_dataset):
        break
    rm.zero_grad() # zeroing out gradient
    x_w, x_l = rm_dataset[i][0].to(device), rm_dataset[i][1].to(device)
    loss = - torch.log(torch.exp(rm(x_w)) / (torch.exp(rm(x_w)) + torch.exp(rm(x_l))))
    loss.backward()
            
    grad_dict={}
    # skip_flag = False
    for k, v in rm.named_parameters():
        if k not in selected_modules:
            continue
        if 'lora_A' in k and 'lm_head' not in k:
            tr_A += 1
            if torch.eq(v.grad, 0).all():
                tr_A_zero += 1
                # print(i, k)
                # skip_flag = True
            grad_dict[k]=v.grad.cpu()
        elif 'lora_B' in k and 'lm_head' not in k:
            tr_B += 1
            # first index of shape indicates low-rank
            if torch.eq(v.grad, 0).all():
                tr_B_zero += 1
            #     print("all 0! tr_grad_dict B")
            grad_dict[k]=v.grad.cpu().T
        elif 'modules_to_save.default.out_proj.weight' in k:
            grad_dict[k]=v.grad.cpu()
        else:
            pass
    tr_grad_dict[i] = grad_dict
    del grad_dict

    # Periodically save the checkpoint
    if i % save_interval == 0 or i == len(rm_dataset) - 1:
        checkpoint = {
            'start_idx': i + 1,  # Save the next index to start from
            'tr_grad_dict': tr_grad_dict,
            'counts': (tr_A, tr_A_zero, tr_B, tr_B_zero)
        }
        with open(grad_path_rm, 'wb') as f:
            pickle.dump(checkpoint, f)

print(len(tr_grad_dict))
print(f"[Train] A: {tr_A_zero} out of {tr_A} are 0; B: {tr_B_zero} out of {tr_B} are 0")
    
val_A_zero = 0
val_A = 0
val_B_zero = 0
val_B = 0
val_grad_dict = {}

grad_path_test = ...
save_interval = 5000  # Number of iterations after which to save

# Check if a previous checkpoint exists
if os.path.exists(grad_path_test):
    with open(grad_path_test, 'rb') as f:
        grads_test = pickle.load(f)
    start_idx = grads_test['start_idx']
    val_grad_dict = grads_test['val_grad_dict']
    val_A, val_A_zero, val_B, val_B_zero = grads_test['counts']
else:
    start_idx = 0
    val_grad_dict = {}
    val_A, val_A_zero, val_B, val_B_zero = 0, 0, 0, 0

for i in tqdm(range(start_idx, len(trust_dataset))):
    if start_idx >= len(trust_dataset):
        break
    rm.zero_grad() # zeroing out gradient
    x_w, x_l = trust_dataset[i][0].to(device), trust_dataset[i][1].to(device)
    loss = - torch.log(torch.exp(rm(x_w)) / (torch.exp(rm(x_w)) + torch.exp(rm(x_l))))
    loss.backward()
    
    grad_dict={}
    for k, v in rm.named_parameters():
        if k not in selected_modules:
            continue
        if 'lora_A' in k and 'lm_head' not in k:
            val_A += 1
            if torch.eq(v.grad, 0).all():
                val_A_zero += 1
            #     print("all 0! val_grad_dict A")
            grad_dict[k]=v.grad.cpu()
        elif 'lora_B' in k and 'lm_head' not in k:
            val_B += 1
            # first index of shape indicates low-rank
            if torch.eq(v.grad, 0).all():
                val_B_zero += 1
            #     print("all 0! val_grad_dict B")
            grad_dict[k]=v.grad.cpu().T
        elif 'modules_to_save.default.out_proj.weight' in k:
            grad_dict[k]=v.grad.cpu()
        else:
            pass
    val_grad_dict[i] = grad_dict
    del grad_dict

    # Periodically save the checkpoint
    if i % save_interval == 0 or i == len(trust_dataset) - 1:
        checkpoint = {
            'start_idx': i + 1,  # Save the next index to start from
            'val_grad_dict': val_grad_dict,
            'counts': (val_A, val_A_zero, val_B, val_B_zero)
        }
        with open(grad_path_test, 'wb') as f:
            pickle.dump(checkpoint, f)

print(f"[Val] A: {val_A_zero} out of {val_A} are 0; B: {val_B_zero} out of {val_B} are 0")
# print(len(val_grad_dict))
n_train, n_val = len(tr_grad_dict.keys()), len(val_grad_dict.keys())
print(n_train, n_val)


# print(val_grad_dict)

val_grad_avg_dict={}
for i, weight_name in tqdm(enumerate(val_grad_dict[0])):
    val_grad_avg_dict[weight_name] = torch.zeros(val_grad_dict[0][weight_name].shape)
    for val_id in val_grad_dict:
        val_grad_avg_dict[weight_name] += val_grad_dict[val_id][weight_name] / n_val
    # if torch.eq(val_grad_avg_dict[weight_name], 0).all():
    #     print("all 0! val_grad_avg_dict")
print("Finished computing avg val")
# compute hvp proposed
lambda_const_param = 10
start_time = time()
hvp_proposed_dict = {}
for i, weight_name in tqdm(enumerate(val_grad_avg_dict)):
    # lambda_const computation
    S = torch.zeros(len(tr_grad_dict.keys()))
    for tr_id in tr_grad_dict:
        tmp_grad = tr_grad_dict[tr_id][weight_name]
        S[tr_id] = torch.mean(tmp_grad**2)
    lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda
    
    # hvp computation
    hvp = torch.zeros(val_grad_avg_dict[weight_name].shape)
    for tr_id in tr_grad_dict:
        tmp_grad = tr_grad_dict[tr_id][weight_name]
        nom = torch.sum(val_grad_avg_dict[weight_name] * tmp_grad)
        denom = (lambda_const + torch.sum(tmp_grad**2))
        C_tmp = torch.sum(val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
        hvp += (val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (n_train*lambda_const)
    hvp_proposed_dict[weight_name] = hvp 
    # if torch.isnan(hvp).any():
    #     print("nan occurs! hvp")
    # print(torch.isnan(hvp).any())

print("Finished computing hvp")
if_tmp_dict = {}
for i, tr_id in tqdm(enumerate(tr_grad_dict)):
    if_tmp_value = 0
    for weight_name in val_grad_avg_dict:
        # if torch.isnan(hvp_proposed_dict[weight_name] * tr_grad_dict[tr_id][weight_name]).any():
        #     print("nan occurs! dot product")
        if_tmp_value += torch.sum(hvp_proposed_dict[weight_name] * tr_grad_dict[tr_id][weight_name])
    if_tmp_dict[tr_id]= -if_tmp_value 
print("Finished computing influence scores")
# print(if_tmp_dict)
scores = pd.Series(if_tmp_dict, dtype=float).to_numpy() 
# print(scores.shape)
# print(scores)

# final_scores = 2 * (scores - np.min(scores)) / (np.max(scores) - np.min(scores)) - 1
normalized_scores = normalize_infl(scores)
print(normalized_scores)
np.save(..., normalized_scores)