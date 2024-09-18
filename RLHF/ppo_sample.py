# Reference : EleutharAI PPO code scripts

import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
# import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
# from tritonclient.utils import np_to_triton_dtype
from sklearn.model_selection import train_test_split

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import wandb
wandb.login()


def get_default_config():
    default_config = TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=10000,
            total_steps=10000,
            batch_size=32, #4,
            checkpoint_interval=10000,
            eval_interval=500,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            entity_name = 'eleutherai',
            project_name = 'pythia-rlhf',
            save_best = False
        ),
        model=ModelConfig(model_path="EleutherAI/pythia-70m", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-70m", truncation_side="left", padding_side = "left", ),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1e-6)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=64,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.05,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="running",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=128,
                top_k=30,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )
    return default_config

def get_config(model_size, default_config):
    config_name = model_size #os.environ.get("CONFIG_NAME", "70M")
    if config_name == "70M":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/do2vbz2o
        default_config.train.batch_size = 32 #8
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 10000 #750
        default_config.model.model_path = "lomahony/eleuther-pythia70m-hh-sft"
        default_config.model.num_layers_unfrozen = 4
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-70m/"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-70m"
        default_config.optimizer.kwargs["lr"] = 5.45e-6
        default_config.optimizer.kwargs["weight_decay"] = 0.0006
        default_config.scheduler.kwargs["eta_min"] = 5.45e-6
        default_config.method.num_rollouts = 32
        default_config.method.target = 5.71
        default_config.method.ppo_epochs = 8
        default_config.method.chunk_size = 4
        default_config.train.group_name = "EleutherAI/pythia-70m-ppo"
    elif config_name == "160M":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/jubaluv8
        default_config.train.batch_size = 8
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 10000
        default_config.model.model_path = "lomahony/eleuther-pythia160m-hh-sft"
        default_config.model.num_layers_unfrozen = 4
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-160m/"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-160m"
        default_config.optimizer.kwargs["lr"] = 1.7e-6
        default_config.optimizer.kwargs["weight_decay"] = 3.81e-5
        default_config.scheduler.kwargs["eta_min"] = 1.7e-6
        default_config.method.num_rollouts = 48
        default_config.method.chunk_size = 4
        default_config.method.ppo_epochs = 6
        default_config.method.target = 6.42
        default_config.train.group_name = "EleutherAI/pythia-160m"
    elif config_name == "410M":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/vpuhppgx
        default_config.train.batch_size = 8
        default_config.train.total_steps = 750
        default_config.train.seq_length = 10000
        default_config.model.num_layers_unfrozen = 3
        default_config.optimizer.kwargs["lr"] = 2.2e-7
        default_config.scheduler.kwargs["eta_min"] = 2.2e-7
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-410m"
        default_config.model.model_path = "lomahony/eleuther-pythia410m-hh-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-410m"
        default_config.method.chunk_size = 4
        default_config.method.num_rollouts = 48
        default_config.method.ppo_epochs = 5
        default_config.method.target = 4.9
        default_config.train.group_name = "EleutherAI/pythia-410m"
    elif config_name == "1B":
        default_config.train.batch_size = 4
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 2500
        default_config.optimizer.kwargs["lr"] = 6e-6
        default_config.optimizer.kwargs["weight_decay"] = 0.0002
        default_config.scheduler.kwargs["eta_min"] = 6e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_1B"
        default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
        default_config.method.chunk_size = 16
        default_config.train.group_name = default_config.model.model_path
    elif config_name == "1.4B":
        default_config.train.batch_size = 4
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 1500
        default_config.model.num_layers_unfrozen = 2
        default_config.optimizer.kwargs["lr"] = 6e-6
        default_config.scheduler.kwargs["eta_min"] = 6e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-1.4b"
        default_config.model.model_path = "lomahony/eleuther-pythia1.4b-hh-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-2.8b"
        default_config.method.chunk_size = 4
        default_config.method.num_rollouts = 48
        default_config.method.ppo_epochs = 8
        default_config.method.target = 5.067
        default_config.train.group_name = "EleutherAI/pythia-1.4b-ppo-hptuning"
    elif config_name == "2.8B":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/9ui2aa0x
        default_config.train.batch_size = 2
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 3000
        default_config.model.num_layers_unfrozen = 19
        default_config.optimizer.kwargs["lr"] = 2.3e-6
        default_config.optimizer.kwargs["weight_decay"] = 1.7e-3
        default_config.scheduler.kwargs["eta_min"] = 2.3e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-2.8b"
        default_config.model.model_path = "lomahony/eleuther-pythia2.8b-hh-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-2.8b"
        default_config.method.chunk_size = 4
        default_config.method.num_rollouts = 48
        default_config.method.ppo_epochs = 4
        default_config.method.target = 5
        default_config.train.group_name = "EleutherAI/pythia-2.8b-ppo"
    elif config_name == "6.9B":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/2uue7ywj
        default_config.train.batch_size = 1
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 6000
        default_config.train.checkpoint_dir = "/fsx/orz/pythia-rlhf-checkpoints/ppo_hh/pythia-6.9b"
        default_config.model.model_path = "lomahony/eleuther-pythia6.9b-hh-sft"
        default_config.model.num_layers_unfrozen = 2
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-6.9b"
        default_config.optimizer.kwargs["lr"] = 3.32e-6
        default_config.scheduler.kwargs["eta_min"] = 1e-6
        default_config.method.num_rollouts = 48
        default_config.method.chunk_size = 4
        default_config.method.ppo_epochs = 3
        default_config.method.target = 5.067
        default_config.train.group_name = "EleutherAI/pythia-6.9b-ppo"
    elif config_name == "12B":
        default_config.train.batch_size = 1
        default_config.train.seq_length = 768
        default_config.train.total_steps = 5000
        default_config.optimizer.kwargs["lr"] = 1e-6
        default_config.scheduler.kwargs["eta_min"] = 1e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_pythia_12b"
        default_config.model.model_path = "EleutherAI/pythia-12b"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-12b"
        default_config.method.num_rollouts = 32
        default_config.method.chunk_size = 8
        default_config.train.group_name = default_config.model.model_path
    else:
        raise ValueError(f"Config {config_name} does not exist."
        "Please append this config into the file before calling it")



def create_reward_fn(model_name):  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
    #reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"  # TODO check

       
    class RewardModel(nn.Module):
        def __init__(self, checkpoint_path, eos_token_id):
            super().__init__()
            # model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            self.model = model
            self.model.config.pad_token_id = reward_tokenizer.pad_token_id

        def forward(self, input_ids):
            #returns = self.model(input_ids)[0]
            returns = -self.model(input_ids).logits
            return returns

    reward_model = RewardModel(model_name, reward_tokenizer.eos_token_id)

    # Load in the parameters
    # NOTE check: we don't need this, correct?
    # directory = snapshot_download("Dahoas/gptj-rm-static", revision="676bfd4d")
    # for fpath in os.listdir(directory):
    #     if fpath.endswith(".pt") or fpath.endswith(".bin"):
    #         checkpoint = os.path.join(directory, fpath)
    #         break
    # reward_model.load_state_dict(torch.load(checkpoint), strict=False)


    reward_model.eval()
    reward_model.requires_grad_(False)
    # TODO undo when switching to GPUs
    reward_device = torch.cuda.device_count() - 1
    # reward_model = reward_model.half().to(reward_device)
    reward_model = reward_model.half().to(reward_device)
    reward_batch_size = 24  # NOTE: this is effectively ignored since the training batch size is 4
    delta_reward = True

    def get_reward(samples):
        input = reward_tokenizer(
            samples,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(reward_device)
        # tokens_good = tokenizer(prompt, response_good,
        #         truncation=True,
        #         max_length=512,
        #         return_token_type_ids=False,
        #         return_tensors="pt",
        #         return_attention_mask=True)
        mbs = reward_batch_size
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = input.input_ids[batch_ixs]
            rewards = -reward_model(input_ids)
            out.extend(rewards)
        return torch.hstack(out)

    def reward_fn(samples, prompts, original_output, **kwargs):
        samples = [s + reward_tokenizer.eos_token for s in samples]
        rewards = get_reward(samples)

        if not delta_reward:
            return rewards

        original_samples = [p + o + reward_tokenizer.eos_token for p, o in zip(prompts, original_output)]
        original_rewards = get_reward(original_samples)
        return rewards - original_rewards

    return reward_fn


def main(default_config, hparams={}):
    print(f"Config : {default_config}")
    trlx.logging.set_verbosity(trlx.logging.INFO)
    config = TRLConfig.update(default_config, hparams)

    # NOTE: changed to toxicity dataset
    dataset = load_dataset("allenai/real-toxicity-prompts")  # NOTE doesn't have test split; doing it ourselves
    #dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic")
    all_prompts = [{"prompt": x["prompt"]["text"], "original_output": x["continuation"]["text"]} for x in dataset["train"] if x['challenging']]
    # breakpoint()
    prompts, eval_prompts = train_test_split(all_prompts, test_size=0.2, random_state=0)
    # eval_prompts = [{"prompt": x["prompt"], "original_output": x["continuation"]} for x in islice(dataset["test"], 35)]  # what is islice doing?
    model_path = "skrishna/pythia-70m-toxicity-model"
    reward_fn = create_reward_fn(model_path)

    # breakpoint()
    trainer, eval_stats = trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],  # TODO: do these need to change?
    )
    if trainer.accelerator.is_main_process:
        trainer.accelerator.print("\n"*100)
        trainer.accelerator.print(eval_stats["reward/mean"])

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    if len(sys.argv) > 2:
        default_config.optimizer.kwargs['lr'] = float(sys.argv[2])
        default_config.scheduler.kwargs['eta_min'] = float(sys.argv[2])
        default_config.optimizer.kwargs['weight_decay'] = float(sys.argv[3])
    # main(hparams)
    for model_size in ['70M', '160M', '410M']:
        wandb.init(project='rlhf_irl')
        default_config = get_default_config()
        get_config(model_size, default_config)
        main(default_config, hparams)