import torch
import torch.nn as nn
import numpy as np
import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
import json
from tqdm import tqdm
import bitsandbytes as bnb
from peft.tuners.lora import QuantLinear
from peft import get_peft_model, LoraConfig
from safetensors.torch import save_file


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path) # , torch_dtype=torch.float16
        self.transformer = model.transformer
        self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        states = self.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        returns = rewards[:, -1]
        return returns

def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names.append(name)
    return names

def _low_rank_decomposition(weight, reduced_rank=16):
    """
    Decompose a 2D matrix into low-rank matrices A and B using SVD.a

    :param weight: The matrix to decompose, of shape (H, W)
    :param reduced_rank: The final rank of the decomposition
    :return: A tuple of tensors (A, B)
    """
    if weight.dim() != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {weight.dim()} dimensions.")

    # SVD Decomposition
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Truncated matrices
    A = Vh[:reduced_rank, :]
    B = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])

    return A, B

def decompose_delta_weight(new_weight, base_weight, alpha, reduced_rank, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    new_weight = new_weight.to(device)
    base_weight = base_weight.to(device)

    """
    Decompose the delta weight into low-rank matrices A and B, considering the alpha scaling factor.

    :param new_weight: The updated weight matrix after applying LoRA.
    :param base_weight: The original weight matrix before LoRA.
    :param alpha: The alpha scaling factor used in LoRA.
    :param reduced_rank: The rank for the low-rank decomposition.
    :return: A tuple of tensors (A, B)
    """
    delta_weight = new_weight - base_weight

    # Check if alpha is applied uniformly
    # Adjust the implementation if alpha is applied differently
    adjusted_delta_weight = delta_weight / alpha

    A, B = _low_rank_decomposition(adjusted_delta_weight, reduced_rank=reduced_rank)

    return A, B

def get_module_peft_name(module_name):
    return module_name.split('.')[-1]

rm_name = ...
rm_cache_dir = ...

rm_tokenizer = AutoTokenizer.from_pretrained(rm_name, cache_dir=rm_cache_dir)
rm_tokenizer.pad_token = rm_tokenizer.eos_token

rm = RewardModel(..., rm_tokenizer.eos_token_id)
directory = snapshot_download(...)
for fpath in os.listdir(directory):
    if fpath.endswith(".pt") or fpath.endswith(".bin"):
        checkpoint = os.path.join(directory, fpath)
        break

rm.load_state_dict(torch.load(checkpoint), strict=False)
rm.eval()
# rm = rm.half()

base_model = AutoModelForCausalLM.from_pretrained(rm_name)
target_model = rm

linear_module_names = find_all_linear_names(base_model)

print(len(linear_module_names))

loras = {

}

# lower rank captures less of the original model, a rank of 32 is probably reasonable for small delta (task specific finetunes and such)
alpha = 1
rank = 4
# print(target_model.state_dict())
for module in tqdm(linear_module_names):
    if module == "lm_head":
        continue
    target_tensor = target_model.state_dict()[module+".weight"]
    base_tensor = base_model.state_dict()[module+".weight"]

    lora_A, lora_B = decompose_delta_weight(target_tensor, base_tensor, alpha, rank)
    loras[f"base_model.model.{module}.lora_A.weight"] = lora_A.to('cpu')
    loras[f"base_model.model.{module}.lora_B.weight"] = lora_B.to('cpu')
loras["v_head"] = target_model.state_dict()["v_head.weight"].to('cpu')
print(f"Number of extracted modules: {len(loras.keys())}")

LORA_OUT_DIR = ...

lora_config = LoraConfig(
        lora_alpha=4, # Setting the alpha to the to decomposition rank value (instead of alpha value used) seems to give better performance. Further testing would be needed to understand what is the optimal alpha value to use
        lora_dropout=0,
        r=4,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= list(set([get_module_peft_name(e) for e in linear_module_names])),
)
peft_model = get_peft_model(base_model, lora_config)

# Save to disk
peft_model.save_pretrained(LORA_OUT_DIR)
del peft_model

for key in loras.keys():
    loras[key] = loras[key].to('cpu').contiguous()

save_file(loras, os.path.join(LORA_OUT_DIR, 'adapter_model.safetensors'))