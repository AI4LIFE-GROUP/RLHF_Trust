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

def DPO_loss(pi_theta, pi_sft, x, y_w, y_l, beta=1.0):
    """
    Compute the DPO loss.

    Args:
    - pi_theta: The current model (parameterized by theta).
    - pi_sft: The supervised fine-tuned model (SFT).
    - x: The context input sequence.
    - y_w: The preferred output sequence.
    - y_l: The non-preferred output sequence.
    - beta: The scaling factor for the log-ratio term (default: 1.0).

    Returns:
    - loss: The DPO loss.
    """
    
    # Compute the log probabilities from the current model for y_w and y_j
    log_pi_theta_y_w = torch.log_softmax(pi_theta(x, y_w), dim=-1)
    log_pi_theta_y_l = torch.log_softmax(pi_theta(x, y_l), dim=-1)
    
    # Compute the log probabilities from the SFT model for y_w and y_j
    log_pi_sft_y_w = torch.log_softmax(pi_sft(x, y_w), dim=-1)
    log_pi_sft_y_l = torch.log_softmax(pi_sft(x, y_l), dim=-1)
    
    # Compute the log-ratio terms for the preferred and non-preferred sequences
    log_ratio_w = beta * (log_pi_theta_y_w - log_pi_sft_y_w)
    log_ratio_l = beta * (log_pi_theta_y_l - log_pi_sft_y_l)
    
    # Compute the difference of the log-ratios
    diff_log_ratios = log_ratio_w - log_ratio_l
    
    # Apply the sigmoid function to the difference of the log-ratios
    sigma_diff = torch.sigmoid(diff_log_ratios)
    
    # Compute the final DPO loss as the negative log of the sigmoid
    loss = -torch.log(sigma_diff).mean()  # Averaging the loss over the batch
    
    return loss

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
        
lm_name_0 = ...
lm_cache_dir_0 = ...
config_0 = PeftConfig.from_pretrained(os.path.abspath(LORA_OUT_DIR_0))
lora_base_0 = AutoModelForCausalLM.from_pretrained(lm_name_0, cache_dir=lm_cache_dir_0)
lora_base_0 = PeftModel.from_pretrained(lora_base_0, os.path.abspath(LORA_OUT_DIR_0))
lm_tokenizer = AutoTokenizer.from_pretrained(lm_name_0, cache_dir=lm_cache_dir_0)
lm_tokenizer.pad_token = lm_tokenizer.eos_token
lm_0 = lora_base_0
lm_0 = lm_0.to(device)

lm_name_1 = ...
lm_cache_dir_1 = ...
config_1 = PeftConfig.from_pretrained(os.path.abspath(LORA_OUT_DIR_1))
lora_base_1 = AutoModelForCausalLM.from_pretrained(lm_name_1, cache_dir=lm_cache_dir_1)
lora_base_1 = PeftModel.from_pretrained(lora_base_1, os.path.abspath(LORA_OUT_DIR_1))
lm_1 = lora_base_1
lm_1 = lm_1.to(device)

lm_dataset_raw = load_dataset(...)

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
        prompt = f"Human: {df.iloc[i]["prompt"]}"
        tokenized_prompt = lm_tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids
        
        continuation = f"Assistant: {df.iloc[i]["continuation"]}"
        tokenized_continuation = lm_tokenizer(continuation, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids

        continuation_old = f"Assistant: {df_old.iloc[i]["continuation"]}"
        tokenized_continuation_old = lm_tokenizer(continuation_old, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids

        trust_dataset.append((tokenized_prompt, tokenized_continuation, tokenized_continuation_old))
        if len(trust_dataset) >= trust_data_num:
            break
    if len(trust_dataset) >= trust_data_num:
        break
print("Metric dataset size: ", len(trust_dataset))

lm_dataset = []
lm_data_num = len(lm_dataset_raw)
for i in tqdm(range(lm_data_num)):
    prompt = f"Human: {lm_dataset_raw[i]["prompt"]}"
    tokenized_prompt = lm_tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids

    chosen = lm_dataset_raw[i]["chosen"]
    tokenized_chosen = lm_tokenizer(chosen, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids
    rejected = lm_dataset_raw[i]["rejected"]
    tokenized_rejected = lm_tokenizer(rejected, return_tensors="pt", padding=False, truncation=True, max_length=1024).input_ids
    lm_dataset.append((tokenized_prompt, tokenized_chosen, tokenized_rejected))
print("rlhf dataset size: ", len(lm_dataset))

names = []
for name, param in lm_0.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        names.append(name)
        param.requires_grad = True

# print(len(names), names)
selected_modules = names[288:-2] # decide which layers to include in the attribution
print(len(selected_modules), selected_modules)
lm_0.train()
lm_1.train()

tr_grad_dict = {}
tr_A_zero = 0
tr_A = 0
tr_B_zero = 0
tr_B = 0

grad_path_lm = ...
save_interval = 10000  # Number of iterations after which to save

# Check if a previous checkpoint exists
if os.path.exists(grad_path_lm):
    with open(grad_path_lm, 'rb') as f:
        grads_lm = pickle.load(f)
    start_idx = grads_lm['start_idx']
    tr_grad_dict = grads_lm['tr_grad_dict']
    tr_A, tr_A_zero, tr_B, tr_B_zero = grads_lm['counts']
else:
    start_idx = 0
    tr_grad_dict = {}
    tr_A, tr_A_zero, tr_B, tr_B_zero = 0, 0, 0, 0

for i in tqdm(range(start_idx, len(lm_dataset))):
    if start_idx >= len(lm_dataset):
        break
    lm_0.zero_grad() # zeroing out gradient
    lm_1.zero_grad() # zeroing out gradient
    x, y_w, y_l = lm_dataset[i][0].to(device), lm_dataset[i][1].to(device), lm_dataset[i][2].to(device) 
    
    loss = DPO_loss(lm_1, lm_0, x, y_w, y_l)
    loss.backward()
            
    grad_dict={}
    # skip_flag = False
    for k, v in lm.named_parameters():
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
    if i % save_interval == 0 or i == len(lm_dataset) - 1:
        checkpoint = {
            'start_idx': i + 1,  # Save the next index to start from
            'tr_grad_dict': tr_grad_dict,
            'counts': (tr_A, tr_A_zero, tr_B, tr_B_zero)
        }
        with open(grad_path_lm, 'wb') as f:
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
    lm_0.zero_grad() # zeroing out gradient
    lm_1.zero_grad() # zeroing out gradient
    x, y_w, y_l = trust_dataset[i][0].to(device), trust_dataset[i][1].to(device), trust_dataset[i][2].to(device) 
    
    loss = DPO_loss(lm_1, lm_0, x, y_w, y_l)
    loss.backward()
    
    grad_dict={}
    for k, v in lm.named_parameters():
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