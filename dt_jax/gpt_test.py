from pprint import pprint

import haiku as hk
import jax
from absl.testing import absltest
from gpt import GPT


class GPTTest(absltest.TestCase):
    def test_forward(self):
        config = {
            "vocab_size": 128,
            "n_embd": 32,
            "n_layer": 2,
            "block_size": 10,
            "embd_pdrop": 0.1,
            "transformer_config": {
                "n_embd": 32,
                "attn_config": {
                    "n_layer": 2,
                    "n_head": 2,
                    "n_embd": 32,
                    "block_size": 10,
                    "attn_pdrop": 0.1,
                    "resid_pdrop": 0.1,
                    "name": "attn",
                },
                "resid_pdrop": 0.1,
            },
        }

        def _fwd(x, is_training):
            model = GPT(**config)
            return model(x, is_training)

        fwd = hk.transform_with_state(_fwd)
        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2, key3 = jax.random.split(key, 3)
        x = jax.random.randint(key1, (3,), 0, 128)
        params, state = fwd.init(key2, x, is_training=True)
        out, state = fwd.apply(params, state, x=x, rng=key3, is_training=True)
        pprint(out)
        self.assertEqual(out.shape, (3, 128))


if __name__ == "__main__":
    absltest.main()
