# flake8: noqa
import collections
import logging
import os
from concurrent import futures

import numpy as np
import tensorflow.compat.v1 as tf
from dopamine.replay_memory import circular_replay_buffer
from torch.utils.data import Dataset

gfile = tf.gfile

STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX
tf.config.experimental.set_visible_devices([], "GPU")


def get_dataset(ds_file_path, n_buffers, n_steps, env_name, data_dir_prefix, trajectories_per_buffer, context_len):
    is_ds_saved = os.path.exists(ds_file_path)
    if is_ds_saved:
        logging.info(f"Loading dataset from {ds_file_path}")
        ds_file = np.load(ds_file_path)
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
        logging.info(f"Creating dataset {ds_file_path}")
        obss, actions, returns, done_idxs, rtgs, timesteps = create_offline_atari_dataset(
            n_buffers, n_steps, env_name, data_dir_prefix, trajectories_per_buffer
        )
        os.makedirs(os.path.dirname(ds_file_path), exist_ok=True)
        with open(ds_file_path, "wb") as ds_file:
            np.savez(
                ds_file,
                obss=obss,
                actions=actions,
                returns=returns,
                done_idxs=done_idxs,
                rtgs=rtgs,
                timesteps=timesteps,
            )
        logging.info(f"Saved dataset to {ds_file_path}")

    return StateActionReturnDataset(obss, context_len * 3, actions, done_idxs, rtgs, timesteps)


class FixedReplayBuffer:
    """
    Object composed of a list of OutofGraphReplayBuffers.
    Adapted from https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/replay_memory/fixed_replay_buffer.py
    """

    def __init__(self, data_dir, replay_suffix, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Initialize the FixedReplayBuffer class.
        Args:
          data_dir: str, log Directory from which to load the replay buffer.
          replay_suffix: int, If not None, then only load the replay buffer
            corresponding to the specific suffix in data directory.
          *args: Arbitrary extra arguments.
          **kwargs: Arbitrary keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs
        self._data_dir = data_dir
        self._loaded_buffers = False
        self.add_count = np.array(0)
        self._replay_suffix = replay_suffix
        if not self._loaded_buffers:
            if replay_suffix is not None:
                assert replay_suffix >= 0, "Please pass a non-negative replay suffix"
                self.load_single_buffer(replay_suffix)
            else:
                self._load_replay_buffers(num_buffers=50)

    def load_single_buffer(self, suffix):
        """Load a single replay buffer."""
        replay_buffer = self._load_buffer(suffix)
        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]
            self.add_count = replay_buffer.add_count
            self._num_replay_buffers = 1
            self._loaded_buffers = True

    def _load_buffer(self, suffix):
        """Loads a OutOfGraphReplayBuffer replay buffer."""
        try:
            # pytype: disable=attribute-error
            replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(*self._args, **self._kwargs)
            replay_buffer.load(self._data_dir, suffix)
            tf.logging.info(f"Loaded replay buffer ckpt {suffix} from {self._data_dir}")
            # pytype: enable=attribute-error
            return replay_buffer
        except tf.errors.NotFoundError:
            return None

    def _load_replay_buffers(self, num_buffers=None):
        """Loads multiple checkpoints into a list of replay buffers."""
        if not self._loaded_buffers:  # pytype: disable=attribute-error
            ckpts = gfile.ListDirectory(self._data_dir)  # pytype: disable=attribute-error
            # Assumes that the checkpoints are saved in a format CKPT_NAME.{SUFFIX}.gz
            ckpt_counters = collections.Counter([name.split(".")[-2] for name in ckpts])
            # Should contain the files for add_count, action, observation, reward,
            # terminal and invalid_range
            ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
            if num_buffers is not None:
                ckpt_suffixes = np.random.choice(ckpt_suffixes, num_buffers, replace=False)
            self._replay_buffers = []
            # Load the replay buffers in parallel
            with futures.ThreadPoolExecutor(max_workers=num_buffers) as thread_pool_executor:
                replay_futures = [thread_pool_executor.submit(self._load_buffer, suffix) for suffix in ckpt_suffixes]
            for f in replay_futures:
                replay_buffer = f.result()
                if replay_buffer is not None:
                    self._replay_buffers.append(replay_buffer)
                    self.add_count = max(replay_buffer.add_count, self.add_count)
            self._num_replay_buffers = len(self._replay_buffers)
            if self._num_replay_buffers:
                self._loaded_buffers = True

    def get_transition_elements(self):
        return self._replay_buffers[0].get_transition_elements()

    def sample_transition_batch(self, batch_size=None, indices=None):
        buffer_index = np.random.randint(self._num_replay_buffers)
        return self._replay_buffers[buffer_index].sample_transition_batch(batch_size=batch_size, indices=indices)

    def load(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    def reload_buffer(self, num_buffers=None):
        self._loaded_buffers = False
        self._load_replay_buffers(num_buffers)

    def save(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    def add(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


# Below are adapted from https://github.com/kzl/decision-transformer/


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1  # +1 for padding
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = np.array(self.data[idx:done_idx], dtype=np.float32).reshape(block_size, -1)  # (block_size, 4*84*84)
        states = states / 255.0
        actions = np.array(self.actions[idx:done_idx], dtype=np.int64)[..., np.newaxis]  # (block_size, 1)
        rtgs = np.array(self.rtgs[idx:done_idx], dtype=np.float32)[..., np.newaxis]
        timesteps = np.array(self.timesteps[idx : idx + 1], dtype=np.int64)

        return states, actions, rtgs, timesteps


def create_offline_atari_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print("loading from buffer %d which has %d already loaded" % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + "/1/replay_logs",
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000,
        )
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                (
                    states,
                    ac,
                    ret,
                    next_states,
                    next_action,
                    next_reward,
                    terminal,
                    indices,
                ) = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += trajectories_per_buffer - trajectories_to_load
            transitions_per_buffer[buffer_num] = i
        print(
            "this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories"
            % (i, len(obss), num_trajectories)
        )

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i - 1, start_index - 1, -1):  # start from i-1
            rtg_j = curr_traj_returns[j - start_index : i - start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print("max rtg is %d" % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions) + 1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index : i + 1] = np.arange(i + 1 - start_index)
        start_index = i + 1
    print("max timestep is %d" % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps
