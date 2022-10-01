import test_with_cpu  # noqa: F401  # isort:skip
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import jit, vmap
from utils import sample, top_k_logits


class TopKLogitsTest(parameterized.TestCase):
    def test_top_k_logits(self):
        logits = jnp.array([-1.0, 2.0, 1.0, 3.0])

        exp_result = jnp.array([float("-inf"), 2.0, float("-inf"), 3.0])
        result = top_k_logits(logits, k=2)
        self.assertTrue(np.allclose(result, exp_result))

    @parameterized.named_parameters(
        (
            "Repated sample",
            jnp.expand_dims(jnp.array([-1.0, 2.0, 1.0, 3.0]), 0).repeat(32, axis=0),
            jnp.expand_dims(jnp.array([float("-inf"), 2.0, float("-inf"), 3.0]), 0).repeat(32, axis=0),
        ),
        (
            "Not repated sample",
            jnp.array([[-1.0, 2.0, 1.0, 3.0], [2.0, 1.0, 10.0, 9.0]]),
            jnp.array([[float("-inf"), 2.0, float("-inf"), 3.0], [float("-inf"), float("-inf"), 10.0, 9.0]]),
        ),
    )
    def test_batch_top_k_logits(self, logits, exp_result):
        batch_top_k_logits = jit(vmap(top_k_logits, (0, None)), static_argnums=1)
        result = batch_top_k_logits(logits, 2)
        self.assertTrue(np.allclose(result, exp_result))


class SampleTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = mock.Mock()
        self.model.return_value = jnp.array(
            [
                [-4.0, 3.0, 2.0, 100.0],
                [-4.0, 3.0, 112.0, 100.0],
                [-4.0, 333.0, 233.0, 100.0],
                [-4.0, -3.0, 2.0, 100.0],
            ]
        )
        self.params = {"w": jnp.array([1.0, 2.0, 3.0, 4.0])}
        self.x = jnp.array([1.0, 2.0, 3.0, 4.0])
        self.rtgs = jnp.array([32, 20, 10, 8])
        self.block_size = 3
        self.exp_result = jnp.array([3], dtype=jnp.int32)

    def test_sample_deterministic(self):
        params, model, x, block_size, rtgs, exp_result = (
            self.params,
            self.model,
            self.x,
            self.block_size,
            self.rtgs,
            self.exp_result,
        )
        result = sample(params, None, model, x, block_size, rtgs=rtgs)
        self.assertTrue(np.array_equal(result, exp_result))

    def test_sample_stochastic(self):
        params, model, x, block_size, rtgs, exp_result = (
            self.params,
            self.model,
            self.x,
            self.block_size,
            self.rtgs,
            self.exp_result,
        )
        result = sample(params, jax.random.PRNGKey(0), model, x, block_size, rtgs=rtgs, sample=True, top_k=2)
        self.assertTrue(np.array_equal(result, exp_result))

    def test_sample_temperature(self):
        params, model, x, block_size, rtgs, exp_result = (
            self.params,
            self.model,
            self.x,
            self.block_size,
            self.rtgs,
            self.exp_result,
        )
        result = sample(
            params,
            jax.random.PRNGKey(0),
            model,
            x,
            block_size,
            rtgs=rtgs,
            sample=True,
            top_k=2,
            temperature=100000,
        )
        self.assertFalse(np.array_equal(result, exp_result))


if __name__ == "__main__":
    absltest.main()
