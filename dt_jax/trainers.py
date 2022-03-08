import math
from typing import Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import optax
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
    rng = jax.random.PRNGKey(42)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


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

                loss, params, opt_state = update(params, subkey, xs, ys, ys, rs, ts, opt_state)
                loss_sum += loss
                it += 1
                pbar.set_description(f"epoch: {epoch+1} iter: {it}.")

            return params, opt_state, loss_sum / len(loader), it

        it = 0
        for epoch in range(config.max_epochs):
            params, opt_state, loss, it = run_epoch(params, opt_state, it)
            print(f"epoch: {epoch}, loss: {loss}")

        return params, opt_state
