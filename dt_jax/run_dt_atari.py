import logging

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

import os
from dataclasses import asdict

from absl import app, flags, logging  # type: ignore
from datasets import get_dataset
from gpt import GPT
from jax.config import config
from trainers import AtariTrainer, AtariTrainerConfig
from utils import set_global_seed

import wandb

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
flags.DEFINE_string("sampled_dataset_path", "./dataset_saved/", "Path to sampled dataset")
flags.DEFINE_float("lr", 6e-4, "Learning rate")
flags.DEFINE_float("beta1", 0.9, "Beta 1 for Adam optimizer")
flags.DEFINE_float("beta2", 0.95, "Beta 2 for Adam optimizer")
flags.DEFINE_string(
    "checkpoint_path",
    "no_checkpoint",
    help="Checkpoint path. wandb flag should be True. wandb run name will be used as checkpoint name.",
)
flags.DEFINE_bool("wandb", False, "Log to wandb")
flags.DEFINE_multi_string("wandb_tags", [], help="wandb tags")
flags.DEFINE_bool("wandb_logging_grads", False, "Log gradients to wandb")

FLAGS = flags.FLAGS


def main(_):
    set_global_seed(FLAGS.seed, pytorch=True)

    ds_file_name = (
        f"{FLAGS.env_name}/n_buffer{FLAGS.n_buffers}"
        f"_n_step{FLAGS.n_steps}_traj_per_buffer{FLAGS.trajectories_per_buffer}_seed{FLAGS.seed}"
    )
    ds_file_path = os.path.join(FLAGS.sampled_dataset_path, ds_file_name)
    train_dataset = get_dataset(
        ds_file_path,
        FLAGS.n_buffers,
        FLAGS.n_steps,
        FLAGS.env_name,
        FLAGS.data_dir_prefix,
        FLAGS.trajectories_per_buffer,
        FLAGS.context_len,
    )
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
        "max_timestep": max(train_dataset.timesteps),
        "model_type": "reward_conditioned",
        "name": "gpt",
    }

    tconf = AtariTrainerConfig(
        max_epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.lr,
        betas=(FLAGS.beta1, FLAGS.beta2),
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * FLAGS.context_len * 3,
        num_workers=4,
        seed=FLAGS.seed,
        model_type=FLAGS.model_type,
        game=FLAGS.env_name,
        max_timestep=max(train_dataset.timesteps),
    )

    if FLAGS.wandb:
        # TODO(yun-kwak): Make types of config consistent
        wandb.init(project="dt-jax", config={"tconf": asdict(tconf), "mconf": mconf}, tags=FLAGS.wandb_tags)
        wandb.config.update(FLAGS)

    def _fwd(states, actions, rtgs, timestep, is_training):
        model = GPT(**mconf)
        return model(states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=is_training)

    trainer = AtariTrainer(_fwd, train_dataset, tconf)
    params = trainer.init_params()
    params, _ = trainer.train(params)


if __name__ == "__main__":
    config.config_with_absl()
    app.run(main)
