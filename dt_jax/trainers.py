# flake8: noqa
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import logging
from optax import chain, clip_by_global_norm, scale, scale_by_adam
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
        logging.info("number of parameters: %d", sum([leave.size for leave in jax.tree_leaves(params)]))
        return params

    def train(self, params, opt_state=None):
        config = self.config
        optimizer = chain(
            clip_by_global_norm(config.grad_norm_clip),
            scale_by_adam(*config.betas),
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

        def run_epoch(params, opt_state):
            loader = DataLoader(
                self.train_ds,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = tqdm(loader, total=len(loader))
            for batch in pbar:
                xs, ys, rs, ts = map(jnp.array, batch)

                # different rng on each device
                config.rng, subkey = jax.random.split(config.rng)

                loss, params, opt_state = update(params, subkey, xs, ys, ys, rs, ts, opt_state)
                losses.append(loss)
                pbar.set_description(f"epoch {epoch+1} train loss {loss:.5f}.")

            return params, opt_state, jnp.array(losses).mean()

        for epoch in range(config.max_epochs):
            params, opt_state, loss = run_epoch(params, opt_state)
            print(f"epoch: {epoch}, loss: {loss}")

        return params, opt_state
