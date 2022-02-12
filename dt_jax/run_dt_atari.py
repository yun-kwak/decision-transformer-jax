# flake8: noqa
# type: ignore

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from absl import app, flags, logging  # type: ignore

from .datasets import StateActionReturnDataset, create_offline_atari_dataset
from .gpt import GPT, GPTConfig
from .trainers import AtariTrainer, AtariTrainerConfig
from .utils import set_global_seed

# set up logging
logging.set_verbosity(logging.INFO)

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

    obss, actions, returns, done_idxs, rtgs, timesteps = create_offline_atari_dataset(
        FLAGS.n_buffers, FLAGS.n_steps, FLAGS.env_name, FLAGS.data_dir_prefix, FLAGS.trajectories_per_buffer
    )
    train_dataset = StateActionReturnDataset(obss, FLAGS.context_length * 3, actions, done_idxs, rtgs, timesteps)

    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=FLAGS.n_layer,
        n_head=8,
        n_embd=128,
        model_type=FLAGS.model_type,
        max_timestep=max(timesteps),
    )
    model = GPT(mconf)

    tconf = AtariTrainerConfig(
        max_epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(train_dataset) * FLAGS.context_length * 3,
        num_workers=4,
        seed=FLAGS.seed,
        model_type=FLAGS.model_type,
        game=FLAGS.env_name,
        max_timestep=max(timesteps),
    )
    trainer = AtariTrainer(model, train_dataset, None, tconf)

    trainer.train()


if __name__ == "__main__":
    app.run(main)
