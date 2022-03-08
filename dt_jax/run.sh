#!/bin/env bash

for seed in 123 231 312 42 77
do
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 python -u run_dt_atari.py --seed $seed --context_len 30 --epochs 5 --model_type 'reward_conditioned' --n_steps 500000 --n_buffers 50 --env_name 'Breakout' --batch_size 128 --data_dir_prefix /data/minimal-atari-replay-dataset/dqn/Breakout/ --checkpoint_name $seed'_128_5' > $seed'_128_5.log' 2>&1
done

for seed in 123 231 312 42 77
do
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 python -u run_dt_atari.py --seed $seed --context_len 30 --epochs 5 --model_type 'reward_conditioned' --n_steps 500000 --n_buffers 50 --env_name 'Breakout' --batch_size 256 --data_dir_prefix /data/minimal-atari-replay-dataset/dqn/Breakout/ --checkpoint_name $seed'_256_5' > $seed'_256_5.log' 2>&1
done

for seed in 123 231 312 42 77
do
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 python -u run_dt_atari.py --seed $seed --context_len 30 --epochs 10 --model_type 'reward_conditioned' --n_steps 500000 --n_buffers 50 --env_name 'Breakout' --batch_size 128 --data_dir_prefix /data/minimal-atari-replay-dataset/dqn/Breakout/ --checkpoint_name $seed'_128_10' > $seed'_128_10.log' 2>&1
done

for seed in 123 231 312 42 77
do
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.85 python -u run_dt_atari.py --seed $seed --context_len 30 --epochs 10 --model_type 'reward_conditioned' --n_steps 500000 --n_buffers 50 --env_name 'Breakout' --batch_size 256 --data_dir_prefix /data/minimal-atari-replay-dataset/dqn/Breakout/ --checkpoint_name $seed'_256_10' > $seed'_256_10.log' 2>&1
done
