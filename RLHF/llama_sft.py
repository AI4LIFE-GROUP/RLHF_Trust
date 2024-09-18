import json
import sys
import os
import hydra
from omegaconf import OmegaConf, DictConfig

from datasets import load_dataset
# from ppo_hh import create_reward_fn

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import signal

# Define a function to handle SIGTERM
def sigterm_handler(signum, frame):
    """
    The signal will always provide two function arguments to the handler: 
    signum: the signal number (in this case the value of signal.SIGTERM)
    frame: the current execution frame
    """
    print("Received SIGTERM signal. Gracefully exiting...")
    # Add code here to save files or perform other cleanup tasks
    

# Register the SIGTERM handler
signal.signal(signal.SIGTERM, sigterm_handler)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024, 
        epochs=3, ### 1
        total_steps=10000000, # ~96200ex/(32bs*7gpus)=~430 steps
        batch_size=32, # 
        checkpoint_interval=1000, #
        eval_interval=100, # 1000
        pipeline="PromptPipeline",
        trainer="AccelerateSFTTrainer",
        checkpoint_dir="checkpoints/sft_hh/pythia-70m",
        seed=0,
        project_name = 'sft-pythia',
        group_name = "pythia-70m",
        # run_name
    ),
    model=ModelConfig(model_path="EleutherAI/pythia-70m", num_layers_unfrozen=-1), ##########
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-70m", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)), 
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=100000000, eta_min=1e-6)),
    method=SFTConfig(
        name="sftconfig",
        gen_kwargs=dict(max_new_tokens=128, top_k=20, top_p=1.0, do_sample=True),
    ),
)


def preprocess(sample):
    sample["chosen_sample"] = sample["prompt"] + sample["chosen"]
    return sample


@hydra.main(version_base=None, config_path=f"{os.getcwd()}/conf", config_name="change_config")
def main(config: DictConfig):

    OmegaConf.resolve(config)
    
    final_config = TRLConfig.update(default_config, OmegaConf.to_container(config)) 
    if os.environ['RANK'] == '0': # print once
        print("Final config: ", final_config)

    # https://huggingface.co/datasets/Dahoas/static-hh
    dataset = load_dataset("Dahoas/static-hh").map(preprocess)
    # reward_fn = create_reward_fn()

    trlx.train(
        config=final_config, 
        samples=dataset["train"]["chosen_sample"],
        eval_prompts=dataset["test"]["prompt"][:40], # 280
        # metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)}, 
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )


if __name__ == "__main__":
    main()