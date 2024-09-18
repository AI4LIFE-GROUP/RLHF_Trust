import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn

helpful_base = load_dataset(
    "Anthropic/hh-rlhf",        # Dataset name
    data_dir="helpful-base",            # Specify the helpfulness subset
    cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/datasets"         # Local cache directory
)
# helpful_online = load_dataset(
#     "Anthropic/hh-rlhf",        # Dataset name
#     data_dir="helpful-online",            # Specify the helpfulness subset
#     cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/datasets"         # Local cache directory
# )
# helpful_rs = load_dataset(
#     "Anthropic/hh-rlhf",        # Dataset name
#     data_dir="helpful-rejection-sampled",            # Specify the helpfulness subset
#     cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/datasets"         # Local cache directory
# )
static_dataset = load_dataset(
    "Dahoas/static-hh",        # Dataset name
    # data_dir="helpful-rejection-sampled",            # Specify the helpfulness subset
    cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/datasets"         # Local cache directory
)
sft_dataset = load_dataset(
    "Dahoas/sft-hh-rlhf",        # Dataset name
    # data_dir="helpful-rejection-sampled",            # Specify the helpfulness subset
    cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/datasets"         # Local cache directory
)
rm_dataset = load_dataset(
    "Dahoas/rm-hh-rlhf",        # Dataset name
    # data_dir="helpful-rejection-sampled",            # Specify the helpfulness subset
    cache_dir="/n/holyscratch01/hlakkaraju_lab/Lab/aaronli/datasets"         # Local cache directory
)
# helpful-base: ~44K, helpful-online: ~22K, harmless-base: ~42.5K, helpful-rejection-sampled: ~52K
print(len(static_dataset['train']))
# print(len(static_dataset['test']))
print(len(sft_dataset['train']))
# print(len(sft_dataset['test']))
print(len(rm_dataset['train']))
# print(len(rm_dataset['test']))

def preprocess_text(text):
    return text.strip().lower()

static_set = set(preprocess_text(sample) for sample in static_dataset['train']['prompt'])
sft_set = set(preprocess_text(sample) for sample in sft_dataset['train']['prompt'])
rm_set = set(preprocess_text(sample) for sample in rm_dataset['train']['prompt'])
base_set = set(preprocess_text(sample) for sample in helpful_base['train']['prompt'])
print(len(static_set), len(sft_set), len(rm_set), len(base_set))
# overlap_1 = static_set.intersection(sft_set)
# overlap_2 = static_set.intersection(rm_set)
# overlap_3 = base_set.intersection(sft_set)
# print(overlap_1, overlap_2, overlap_3)