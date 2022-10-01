#!/bin/env bash

for seed in 123 231 312 42 77
do
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 python -u dt_jax/run_dt_atari.py\
    --seed $seed --context_len 30 --epochs 10 --model_type 'reward_conditioned' --n_steps 500000\
    --n_buffers 50 --env_name 'Breakout' --batch_size 128\
    --data_dir_prefix /data/minimal-atari-replay-dataset/\
    --checkpoint_path ./checkpoints/batch128_$seed --wandb\
    > ./logs/batch128_$seed'.log' 2>&1 &&\
    echo "Finished seed $seed"
done
