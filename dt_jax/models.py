# flake8: noqa
from typing import Dict, Mapping, Optional

import haiku as hk
import jax
import jax.numpy as jnp


class RunModel:
    """
    Container for JAX model.
    Adapted from AlphaFold
    https://github.com/deepmind/alphafold/blob/1e216f93f06aa04aa699562f504db1d02c3b704c/alphafold/model/model.py#L48-L66
    """

    def __init__(self, config: Dict, params: Optional[hk.Params] = None):
        self.config = config
        self.params = params

        def _forward_fn(batch):
            model = build_model(self.config.model)
            return model(batch, is_training=False, compute_loss=False, ensemble_representations=True)

        self.apply = jax.jit(hk.transform(_forward_fn).apply)
        self.init = jax.jit(hk.transform(_forward_fn).init)
