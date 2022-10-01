import math
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import flags
from configs import AtariDefaultOptimalReturn
from envs import AtariEnv, AtariEnvConfig
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
from utils import flatten_params, sample

import wandb

FLAGS = flags.FLAGS


@dataclass
class AtariTrainerConfig:
    # optimization parameters
    max_timestep: int
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_norm_clip: float = 1.0
    weight_decay: float = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay: bool = False
    # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    warmup_tokens: int = int(375e6)
    final_tokens: int = int(260e9)  # (at what point we reach 10% of original LR)
    num_workers: int = 0  # for DataLoader
    seed: int = 42
    model_type: str = "reward_conditioned"
    game: str = "Breakout"

    def __post_init__(self):
        self.rng = jax.random.PRNGKey(self.seed)


def cross_entropy(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1]).squeeze()
    loss = -jax.nn.log_softmax(logits) * one_hot
    loss = loss.sum() / one_hot.sum()
    return loss


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
        blacklist = ["emb", "layer_norm", "attn"]
        whitelist = ["linear", "conv"]
        if any([layer in key for layer in blacklist]):
            return 0
        if any([layer in key for layer in whitelist]):
            return 1
        if key == "b":
            return 0
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
    def __init__(self, fwd_fn, train_ds, config):
        self.fwd_fn = fwd_fn
        self.train_ds = train_ds
        self.train_dl = DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers)
        self.config = config
        self.init = hk.transform(fwd_fn).init
        self.apply = hk.transform(fwd_fn).apply

    def init_params(self):
        self.config.rng, subkey = jax.random.split(self.config.rng)
        batch = next(iter(self.train_dl))
        xs, ys, rs, ts = map(jnp.array, batch)
        params = self.init(subkey, xs[0], ys[0], rs[0], ts[0], True)
        num_params = hk.data_structures.tree_size(params)
        byte_size = hk.data_structures.tree_bytes(params)
        print(f"{num_params} params, size: {byte_size / 1e6:.2f}MB")
        return params

    def train(self, params, opt_state=None):
        config = self.config

        lr_sheduler = lr_schedule(config, self.train_ds.block_size)

        optimizer = chain(
            clip_by_global_norm(config.grad_norm_clip),
            scale_by_adam(*config.betas),
            add_decayed_weights(config.weight_decay, configure_decay_mask(params)),
            scale_by_schedule(lr_sheduler),
            scale(-1),
        )
        if opt_state is None:
            opt_state = optimizer.init(params)

        def loss_fn(params, subkey, states, actions, targets, rtgs, timestep):
            logits = jax.vmap(self.apply, in_axes=[None, None, 0, 0, 0, 0, None])(
                params, subkey, states, actions, rtgs, timestep, True
            )  # TODO: Receive subkey and is_training
            loss = cross_entropy(logits, targets)
            return loss

        @jax.jit
        def update(params, subkey, states, actions, targets, rtgs, timestep, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params, subkey, states, actions, targets, rtgs, timestep)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, params, opt_state

        def run_epoch(params, opt_state, it):

            loader = DataLoader(
                self.train_ds,
                shuffle=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            pbar = tqdm(loader, total=len(loader))
            loss_sum = 0
            for batch in pbar:
                xs, ys, rs, ts = map(jnp.array, batch)

                # different rng on each device
                config.rng, subkey = jax.random.split(config.rng)

                loss, params, opt_state, grads = update(params, subkey, xs, ys, ys, rs, ts, opt_state)
                loss_sum += loss
                it += 1

                # Evaluate the model every 100 steps
                if it % 100 == 0:
                    # TODO(yun-kwak): Remove redundant code
                    if self.config.model_type == "naive":
                        eval_mean_return, eval_std_return = self.get_returns(0, params, n_epi=1)
                    elif self.config.model_type == "reward_conditioned":
                        eval_mean_return, eval_std_return = self.get_returns(
                            AtariDefaultOptimalReturn[self.config.game], params, n_epi=1
                        )
                    else:
                        raise NotImplementedError()

                    print(
                        f"Iteration {it}: eval mean return: {eval_mean_return:.2f}, "
                        + f"eval std return: {eval_std_return:.2f}"
                    )
                    if FLAGS.wandb:
                        wandb.log(
                            {"eval_mean_return": eval_mean_return, "eval_std_return": eval_std_return},
                            commit=False,
                        )

                    if FLAGS.checkpoint_path != "no_checkpoint":
                        # Save the checkpoint
                        assert FLAGS.wandb, "FLAGS.wandb must be True to save checkpoints"
                        import pickle

                        pickle.dump(params, open(f"{FLAGS.checkpoint_path}_{wandb.run.name}_iter_{it}.pkl", "wb"))
                if FLAGS.wandb:
                    if FLAGS.wandb_logging_grads:
                        for name, value in flatten_params(grads):
                            wandb.log({f"grad/{name}": wandb.Histogram(value)}, commit=False)
                    wandb.log(
                        {
                            "batch_loss": loss,
                            "samples": it * config.batch_size,
                            "gd_steps": it,
                        },
                    )
                pbar.set_description(f"epoch: {epoch+1} iter: {it}.")

            return params, opt_state, loss_sum / len(loader), it

        it = 0
        for epoch in range(config.max_epochs):
            params, opt_state, loss, it = run_epoch(params, opt_state, it)
            print(f"epoch: {epoch + 1}, loss: {loss}")

            if self.config.model_type == "naive":
                eval_mean_return, eval_std_return = self.get_returns(0, params)
            elif self.config.model_type == "reward_conditioned":
                eval_mean_return, eval_std_return = self.get_returns(
                    AtariDefaultOptimalReturn[self.config.game], params
                )
            else:
                raise NotImplementedError()
            print(f"eval return: {eval_mean_return}, std: {eval_std_return}")
            if FLAGS.wandb:
                wandb.log(
                    {
                        "epoch_train_loss": loss,
                        "epoch_eval_mean_return": eval_mean_return,
                        "epoch_eval_std_return": eval_std_return,
                        "epoch": epoch + 1,
                    },
                    commit=False,
                )

            if FLAGS.checkpoint_path != "no_checkpoint":
                # Save the checkpoint
                assert FLAGS.wandb, "FLAGS.wandb must be True to save checkpoints"
                import pickle

                pickle.dump(params, open(f"{FLAGS.checkpoint_path}_{wandb.run.name}_epoch_{epoch + 1}.pkl", "wb"))

        return params, opt_state

    def get_returns(self, ret, params, n_epi=10):
        # TODO(yun-kwak): Parallel evaluation
        args = AtariEnvConfig(self.config.seed, self.config.game.lower())
        env = AtariEnv(args)
        env.eval()
        T_rewards = []
        done = True
        pbar = tqdm(range(n_epi))
        jit_sample = jax.jit(
            partial(sample, model=self.apply), static_argnames=["block_size", "temperature", "sample"]
        )

        def build_traj_block(states, actions, rtgs, timestep):
            context_len = self.train_ds.block_size // 3
            s = jnp.asarray(states[-context_len:], dtype=jnp.float32)
            a = jnp.asarray(actions[-context_len:], dtype=jnp.int32)
            r = jnp.asarray(rtgs[-context_len:], dtype=jnp.int32)[..., jnp.newaxis]  # rtgs
            t = ((min(timestep, self.config.max_timestep) * jnp.ones((1), dtype=jnp.int32)),)
            return s, a, r, t

        for it in pbar:
            # first state is from env, first rtg is target return, and first timestep is 0
            states = []
            actions = []
            rtgs = [ret]

            state = env.reset()  # (4, 84, 84)
            states.append(state)

            self.config.rng, subkey = jax.random.split(self.config.rng)
            s, _, r, t = build_traj_block(states, actions, rtgs, 1)
            sampled_action = jit_sample(
                params,
                subkey,
                states=s,
                block_size=self.train_ds.block_size,
                temperature=1.0,
                sample=True,
                actions=None,
                rtgs=r,
                timestep=t,
            )
            j = 0
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = np.array(sampled_action)[-1]
                state, reward, done = env.step(action)
                reward_sum += reward
                pbar.set_description(f"eval iter: {it}. reward_sum: {reward_sum}, step: {j}")
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    pbar.set_description(f"eval iter: {it}. mean: {np.mean(T_rewards)}, std: {np.std(T_rewards)}")
                    break

                states.append(state)
                rtgs.append(rtgs[-1] - reward)
                actions.append(sampled_action)

                self.config.rng, subkey = jax.random.split(self.config.rng)
                s, a, r, t = build_traj_block(states, actions, rtgs, j)
                sampled_action = jit_sample(
                    params,
                    subkey,
                    states=s,
                    block_size=self.train_ds.block_size,
                    temperature=1.0,
                    sample=True,
                    actions=a,
                    rtgs=r,
                    timestep=t,
                )

        # env.close()
        return np.mean(T_rewards), np.std(T_rewards)
