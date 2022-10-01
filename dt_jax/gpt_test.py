import test_with_cpu  # noqa: F401  # isort:skip

import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import absltest
from gpt import GPT

config = {
    "vocab_size": 128,
    "n_embd": 32,
    "n_layer": 2,
    "context_len": 30,
    "embd_pdrop": 0.1,
    "transformer_config": {
        "n_embd": 32,
        "attn_config": {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "context_len": 30,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "name": "attn",
        },
        "resid_pdrop": 0.1,
    },
    "max_timestep": 200,
    "model_type": "reward_conditioned",
    "name": "gpt",
}


def _fwd(states, actions, rtgs, timestep, is_training):
    model = GPT(**config)
    return model(states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=is_training)


class GPTTest(absltest.TestCase):
    def test_forward(self):
        fwd = hk.transform(_fwd)

        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)

        context_len = config["context_len"]
        states = jnp.zeros((context_len, 4 * 84 * 84))
        actions = jnp.zeros((context_len, 1), dtype=jnp.int32)
        rtgs = jnp.zeros((context_len, 1))
        timestep = jnp.zeros((1,), dtype=jnp.int32)

        params = fwd.init(key1, states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=True)
        out = fwd.apply(params, key2, states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=True)
        self.assertEqual(out.shape, (context_len, config["vocab_size"]))

    def test_jit(self):
        init = jax.jit(hk.transform(_fwd).init, static_argnames=["is_training"])
        apply = jax.jit(hk.transform(_fwd).apply, static_argnames=["is_training"])

        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)

        context_len = config["context_len"]
        states = jnp.zeros((context_len, 4 * 84 * 84))
        actions = jnp.zeros((context_len, 1), dtype=jnp.int32)
        rtgs = jnp.zeros((context_len, 1))
        timestep = jnp.zeros((1,), dtype=jnp.int32)

        params = init(key1, states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=True)
        out = apply(params, key2, states=states, actions=actions, rtgs=rtgs, timestep=timestep, is_training=True)
        self.assertEqual(out.shape, (context_len, config["vocab_size"]))

    def test_vmap(self):
        init = hk.transform(_fwd).init
        apply = jax.vmap(hk.transform(_fwd).apply, in_axes=[None, None, 0, 0, 0, 0, None])

        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)

        context_len = config["context_len"]
        batch_size = 32
        states = jnp.zeros((batch_size, context_len, 4 * 84 * 84))
        actions = jnp.zeros((batch_size, context_len, 1), dtype=jnp.int32)
        rtgs = jnp.zeros((batch_size, context_len, 1))
        timestep = jnp.zeros(
            (
                batch_size,
                1,
            ),
            dtype=jnp.int32,
        )

        params = init(key1, states[0], actions[0], rtgs[0], timestep[0], True)
        out = apply(
            params, key2, states, actions, rtgs, timestep, True
        )  # Keyword arguments are not supported by vmap. refer to https://github.com/google/jax/issues/7465
        self.assertEqual(out.shape, (batch_size, context_len, config["vocab_size"]))

    def test_jit_vmap(self):
        init = hk.transform(_fwd).init
        apply = jax.jit(jax.vmap(hk.transform(_fwd).apply, in_axes=[None, None, 0, 0, 0, 0, None]), static_argnums=6)

        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)

        context_len = config["context_len"]
        batch_size = 32
        states = jnp.zeros((batch_size, context_len, 4 * 84 * 84))
        actions = jnp.zeros((batch_size, context_len, 1), dtype=jnp.int32)
        rtgs = jnp.zeros((batch_size, context_len, 1))
        timestep = jnp.zeros(
            (
                batch_size,
                1,
            ),
            dtype=jnp.int32,
        )

        params = init(key1, states[0], actions[0], rtgs[0], timestep[0], True)
        out = apply(
            params, key2, states, actions, rtgs, timestep, True
        )  # Keyword arguments are not supported by vmap. refer to https://github.com/google/jax/issues/7465
        self.assertEqual(out.shape, (batch_size, context_len, config["vocab_size"]))


if __name__ == "__main__":
    absltest.main()
