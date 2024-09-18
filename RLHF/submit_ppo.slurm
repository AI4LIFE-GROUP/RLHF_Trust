#!/bin/bash

#SBATCH -c 40        # Number of cores (-c)
#SBATCH -t 2-00:00     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu,gpu  # Partition to submit to
#SBATCH --mem=200000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH -J dpo-2.8b #job_name
#SBATCH -o ./%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./%j.err # File to which STDERR will be written, %j inserts jobid


source activate RLHF
nvidia-smi
accelerate launch --main_process_port 29500 --gradient_accumulation_steps 4 --config_file conf/zero2-bf16.yaml ppo_new.py model.model_size="2.8b" train.batch_size=1 train.checkpoint_interval=800 train.eval_interval=100 optimizer.lr=7e-6 scheduler.kwargs.eta_min=7e-6 train.total_steps=800