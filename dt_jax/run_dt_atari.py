# flake8: noqa
import logging

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

import os

import numpy as np
from absl import app, flags, logging  # type: ignore
from datasets import StateActionReturnDataset, create_offline_atari_dataset
from gpt import GPT
from trainers import AtariTrainer, AtariTrainerConfig
from utils import set_global_seed

flags.DEFINE_integer("seed", 17, "Random seed")
flags.DEFINE_integer("context_len", 30, "Context length")
flags.DEFINE_integer("epochs", 5, "Number of epochs")
flags.DEFINE_string("model_type", "reward_conditioned", "Model type")
flags.DEFINE_integer("n_steps", 500000, "Number of steps")
flags.DEFINE_integer("n_buffers", 50, "Number of buffers")
flags.DEFINE_string("env_name", "Breakout", "Env name")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("n_layer", 6, "Number of layers in Transformer")
flags.DEFINE_integer("trajectories_per_buffer", 10, "Number of trajectories to sample from each of the buffers")
flags.DEFINE_string("data_dir_prefix", "./dqn_replay/", "Data dir prefix")

FLAGS = flags.FLAGS


def main(_):
    set_global_seed(FLAGS.seed, pytorch=True)

    ds_file_name = f"dataset_saved/{FLAGS.env_name}/n_buffer{FLAGS.n_buffers}_n_step{FLAGS.n_steps}_traj_per_buffer{FLAGS.trajectories_per_buffer}_seed{FLAGS.seed}"
    is_ds_saved = os.path.exists(ds_file_name)
    if is_ds_saved:
        logging.info(f"Loading dataset from {ds_file_name}")
        ds_file = np.load(ds_file_name)
        obss, actions, returns, done_idxs, rtgs, timesteps = (
            ds_file["obss"],
            ds_file["actions"],
            ds_file["returns"],
            ds_file["done_idxs"],
            ds_file["rtgs"],
            ds_file["timesteps"],
        )
        logging.info("Loaded dataset from the folder")
    else:
        logging.info("Creating dataset")
        obss, actions, returns, done_idxs, rtgs, timesteps = create_offline_atari_dataset(
            FLAGS.n_buffers, FLAGS.n_steps, FLAGS.env_name, FLAGS.data_dir_prefix, FLAGS.trajectories_per_buffer
        )
        os.makedirs(os.path.dirname(ds_file_name), exist_ok=True)
        with open(ds_file_name, "wb") as ds_file:
            np.savez(
                ds_file,
                obss=obss,
                actions=actions,
                returns=returns,
                done_idxs=done_idxs,
                rtgs=rtgs,
                timesteps=timesteps,
            )
        logging.info(f"Saved dataset to {ds_file_name}")

    train_dataset = StateActionReturnDataset(obss, FLAGS.context_len * 3, actions, done_idxs, rtgs, timesteps)

    mconf = {
        "vocab_size": train_dataset.vocab_size,
        "n_embd": 128,
        "n_layer": 6,
        "context_len": train_dataset.block_size,
        "embd_pdrop": 0.1,
        "transformer_config": {
            "n_embd": 128,
            "attn_config": {
                "n_layer": 6,
                "n_head": 8,
                "n_embd": 128,
                "context_len": train_dataset.block_size,
                "attn_pdrop": 0.1,
                "resid_pdrop": 0.1,
                "name": "attn",
            },
            "resid_pdrop": 0.1,
        },
        "max_timestep": max(timesteps),
        "model_type": "reward_conditioned",
        "name": "gpt",
    }

    tconf = AtariTrainerConfig(
        max_epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * FLAGS.context_len * 3,
        num_workers=4,
        seed=FLAGS.seed,
        model_type=FLAGS.model_type,
        game=FLAGS.env_name,
        max_timestep=max(timesteps),
    )

    def _fwd(states, actions, rtgs, timestep, is_training):
        model = GPT(**mconf)
        return model(states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=is_training)

    trainer = AtariTrainer(_fwd, train_dataset, tconf)
    params = trainer.init_params()
    params, _ = trainer.train(params)


if __name__ == "__main__":
    app.run(main)
