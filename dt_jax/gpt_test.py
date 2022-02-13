import test_with_cpu  # noqa: F401  # isort:skip

import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import absltest
from gpt import GPT


class GPTTest(absltest.TestCase):
    def test_forward(self):
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

        def _fwd(**x):
            model = GPT(**config)
            return model(**x)

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


if __name__ == "__main__":
    absltest.main()
