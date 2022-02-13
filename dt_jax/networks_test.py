import test_with_cpu  # noqa: F401  # isort:skip

import haiku as hk
import jax
import jax.numpy as jnp
from absl.testing import absltest
from networks import CausalSelfAttention, Dropout, TransformerBlock


class DropoutTest(absltest.TestCase):
    def test_forward(self):
        def _fwd(x, is_training):
            dropout = Dropout(pdrop=0.5, name="dropout")
            return dropout(x, is_training=is_training)

        x = jnp.ones((2, 2))
        fwd = hk.transform(_fwd)
        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)
        params = fwd.init(key1, x, is_training=True)
        _ = fwd.apply(params, x=x, rng=key2, is_training=True)


class CausalSelfAttentionTest(absltest.TestCase):
    def test_forward(self):
        def _fwd(x, is_training):
            model = CausalSelfAttention(n_layer=2, n_head=2, n_embd=32, context_len=2)
            return model(x, is_training)

        x = jnp.zeros((3, 32))  # T, n_embd
        fwd = hk.transform(_fwd)
        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)
        params = fwd.init(key1, x, is_training=True)
        out = fwd.apply(params, x=x, rng=key2, is_training=True)
        self.assertEqual(out.shape, (3, 32))


class TransformerBlockTest(absltest.TestCase):
    def test_forward(self):
        attn_config = {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "context_len": 3,
            "attn_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "name": "attn",
        }

        def _fwd(x, is_training):
            block = TransformerBlock(n_embd=32, attn_config=attn_config, resid_pdrop=0.1, name="block")
            return block(x, is_training=is_training)

        x = jnp.ones((3, 32))
        fwd = hk.transform(_fwd)
        seed = 0
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key, 2)
        params = fwd.init(key1, x, is_training=True)
        out = fwd.apply(params, x=x, rng=key2, is_training=True)
        self.assertEqual(out.shape, (3, 32))


if __name__ == "__main__":
    absltest.main()
