# flake8: noqa
# TODO: Implement a trainer for the JAX model.
import math
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
import optax
from absl import logging
from configs import AtariDefaultOptimalReturn
from envs import AtariEnv, AtariEnvConfig
from jax import local_device_count
from optax import (
    add_decayed_weights,
    chain,
    clip_by_global_norm,
    scale,
    scale_by_adam,
    scale_by_schedule,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class AtariTrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader


def lr_schedule(config, step_items):
    def lr_sheduler(nstep):
        # decay the learning rate based on our progress
        n_tokens = jnp.array(nstep, float) * config.batch_size * step_items
        if config.lr_decay:
            progress = (n_tokens - config.warmup_tokens) / max(1, config.final_tokens - config.warmup_tokens)
            lr_mult = jnp.where(
                n_tokens < config.warmup_tokens,
                # linear warmup
                n_tokens / jnp.fmax(1, config.warmup_tokens),
                # cosine learning rate decay
                jnp.fmax(0.1, 0.5 * (1.0 + jnp.cos(math.pi * progress))),
            )
            lr = config.learning_rate * lr_mult
        else:
            lr = config.learning_rate
        return lr

    return lr_sheduler


def pmap_batch(batch):
    """Splits the first axis of `arr` evenly across the number of devices."""
    per_device_batch_size = batch.shape[0] // local_device_count()
    batch = batch[: per_device_batch_size * local_device_count()]  # trim the rest of the batch
    return batch.reshape(local_device_count(), per_device_batch_size, *batch.shape[1:])


def pmap_on(tree):
    return jax.tree_map(lambda x: jnp.array([x] * local_device_count()), tree)


def pmap_off(tree):
    return jax.device_get(jax.tree_map(lambda x: x[0], tree))


def configure_decay_mask(params):
    """
    This function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    """
    # replace when registry is accessible
    # https://github.com/google/jax/blob/97a5719fcb40af7231b5f803f965063538282f8e/jax/_src/tree_util.py#L197
    tree_types = (tuple, list, dict, Mapping, type(None))

    def check_decay_list(key, parent_decays):
        if any([layer in key for layer in ["embeddings", "layer_norm", "multi_head_attention"]]):
            return 0
        if key == "b":
            return 0
        if "linear" in key:
            return 1
        return parent_decays

    def check_decay(item, parent_decays):
        if not isinstance(item, tree_types):
            return parent_decays
        tree_type = type(item)
        if isinstance(item, (dict, Mapping)):
            tree = {k: check_decay(v, check_decay_list(k, parent_decays)) for k, v in item.items()}
        else:
            tree = [check_decay(v, parent_decays) for v in item]
        return tree_type(tree)

    mask = check_decay(params, -1)
    # validate that we considered every parameter
    assert all([decays >= 0 for decays in jax.tree_flatten(mask)[0]])
    return jax.tree_map(lambda x: x == 1, mask)


class AtariTrainer:
    def __init__(self, hk_loss_fn, train_dataset, config):
        self.hk_loss_fn = hk_loss_fn
        self.train_dataset = train_dataset
        self.config = config

    def save_checkpoint(self, params, opt_state):
        if self.config.ckpt_path is None:
            return
        logging.info("saving to %s", self.config.ckpt_path)
        pickle.dump(pmap_off(params), open(self.config.ckpt_path + "/model.npy", "wb"))
        pickle.dump(pmap_off(opt_state), open(self.config.ckpt_path + "/optimizer.npy", "wb"))

    def init_params(self):
        self.config.rng, subkey = jax.random.split(self.config.rng)
        train_dl = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )
        batch = next(iter(train_dl))
        xs, ys, rs, ts = map(jnp.array, batch)
        params = self.hk_loss_fn.init(subkey, xs, ys, rs, ts, True)
        logging.info("number of parameters: %d", sum([leave.size for leave in jax.tree_leaves(params)]))
        return params

    def train(self):
        config = self.config
        lr_sheduler = lr_schedule(
            config, config.step_tokens if config.step_tokens is not None else self.train_dataset.block_size
        )

        optimiser = chain(
            clip_by_global_norm(config.grad_norm_clip),
            scale_by_adam(*config.betas),
            add_decayed_weights(config.weight_decay, configure_decay_mask(params)),
            scale_by_schedule(lr_sheduler),
            scale(-1),
        )
        if opt_state is None:
            opt_state = optimiser.init(params)
        params, opt_state = map(pmap_on, (params, opt_state))
        loss_fn = self.hk_loss_fn.apply

        @partial(jax.pmap, axis_name="num_devices")
        def update(params, subkey, x, y, target, r, t, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params, subkey, x, y, target, r, t)

            grads = jax.lax.pmean(grads, axis_name="num_devices")
            loss = jax.lax.pmean(loss, axis_name="num_devices")

            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        @partial(jax.pmap, axis_name="num_devices")
        def get_loss(params, subkey, xs, ys):
            loss = loss_fn(params, subkey, xs, ys)
            return jax.lax.pmean(loss, axis_name="num_devices")

        def run_epoch(params, opt_state, it):
            loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = tqdm(loader, total=len(loader))
            for batch in pbar:
                xs, ys, rs, ts = map(pmap_batch, map(jnp.array, batch))

                # different rng on each device
                config.rng, *subkey = jax.random.split(config.rng, num=local_device_count() + 1)
                subkey = jnp.array(subkey)

                loss, params, opt_state = update(params, subkey, xs, ys, ys, rs, ts, opt_state)
                loss = loss[0]
                losses.append(loss)
                pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss:.5f}. lr {lr_sheduler(it):e}")

            return params, opt_state, it

        it = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            params, opt_state, it = run_epoch(params, opt_state, it)
            run_epoch("train", epoch_num=epoch)

            # -- pass in target returns
            # if self.config.model_type == "naive":
            #     eval_return = self.get_returns(0)
            # elif self.config.model_type == "reward_conditioned":
            #     eval_return = self.get_returns(AtariDefaultOptimalReturn[self.config.game])
            # else:
            #     raise NotImplementedError()

    # def get_returns(self, ret):
    #     self.model.train(False)
    #     args = AtariEnvConfig(self.config.game.lower(), self.config.seed)
    #     env = AtariEnv(args)
    #     env.eval()

    #     T_rewards, T_Qs = [], []
    #     done = True
    #     for i in range(10):
    #         state = env.reset()
    #         state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
    #         rtgs = [ret]
    #         # first state is from env, first rtg is target return, and first timestep is 0
    #         sampled_action = sample(
    #             self.model.module,
    #             state,
    #             1,
    #             temperature=1.0,
    #             sample=True,
    #             actions=None,
    #             rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
    #             timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
    #         )

    #         j = 0
    #         all_states = state
    #         actions = []
    #         while True:
    #             if done:
    #                 state, reward_sum, done = env.reset(), 0, False
    #             action = sampled_action.cpu().numpy()[0, -1]
    #             actions += [sampled_action]
    #             state, reward, done = env.step(action)
    #             reward_sum += reward
    #             j += 1

    #             if done:
    #                 T_rewards.append(reward_sum)
    #                 break

    #             state = state.unsqueeze(0).unsqueeze(0).to(self.device)

    #             all_states = torch.cat([all_states, state], dim=0)

    #             rtgs += [rtgs[-1] - reward]
    #             # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
    #             # timestep is just current timestep
    #             sampled_action = sample(
    #                 self.model.module,
    #                 all_states.unsqueeze(0),
    #                 1,
    #                 temperature=1.0,
    #                 sample=True,
    #                 actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
    #                 rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
    #                 timesteps=(
    #                     min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)
    #                 ),
    #             )
    #     env.close()
    #     eval_return = sum(T_rewards) / 10.0
    #     print("target return: %d, eval return: %d" % (ret, eval_return))
    #     self.model.train(True)
    #     return eval_return
