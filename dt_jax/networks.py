import math
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class Dropout(hk.Module):
    def __init__(self, pdrop: float, name: Optional[str] = None):
        super().__init__(name)
        self.pdrop = pdrop

    def __call__(self, x, is_training):
        if is_training:
            return hk.dropout(hk.next_rng_key(), self.pdrop, x)
        else:
            return x


class CausalSelfAttention(hk.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        n_embd: int,
        context_len: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        name: Optional[str] = None,
    ):
        assert n_embd % n_head == 0

        super().__init__(name)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.context_len = context_len
        self.max_block_size = context_len * 3
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        # key, query, value projections for all heads
        self.key = hk.Linear(
            n_embd,
            name="key",
            w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
            b_init=hk.initializers.Constant(0.0),
        )
        self.query = hk.Linear(
            n_embd,
            name="query",
            w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
            b_init=hk.initializers.Constant(0.0),
        )
        self.value = hk.Linear(
            n_embd,
            name="value",
            w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
            b_init=hk.initializers.Constant(0.0),
        )
        # regularization
        self.attn_drop = Dropout(attn_pdrop, name="attn_dropout")
        self.resid_drop = Dropout(resid_pdrop, name="resid_dropout")
        # output projection
        self.proj = hk.Linear(
            self.n_embd,
            w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
            b_init=hk.initializers.Constant(0.0),
        )
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = np.tril(np.ones((self.max_block_size + 1, self.max_block_size + 1)))

    def __call__(self, x, is_training):
        T, C = x.shape  # T: tokens, C: channels
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).reshape(T, self.n_head, C // self.n_head).swapaxes(0, 1)  # (nh, T, hs)
        q = self.query(x).reshape(T, self.n_head, C // self.n_head).swapaxes(0, 1)  # (nh, T, hs)
        v = self.value(x).reshape(T, self.n_head, C // self.n_head).swapaxes(0, 1)  # (nh, T, hs)

        # causal self-attention; Self-attend: (nh, T, hs) x (nh, hs, T) -> (nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        att = jnp.where(self.mask[:T, :T], att, float("-inf"))
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_drop(att, is_training=is_training)
        y = att @ v  # (nh, T, T) x (nh, T, hs) -> (nh, T, hs)
        y = y.swapaxes(0, 1).reshape(T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y), is_training=is_training)
        return y


class TransformerBlock(hk.Module):
    def __init__(self, n_embd: int, attn_config: dict, resid_pdrop: float = 0.1, name: Optional[str] = None):
        super().__init__(name)
        self.n_embd = n_embd
        self.attn_config = attn_config
        self.resid_pdrop = resid_pdrop

        # initialize submodules
        self.ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.attn = CausalSelfAttention(**self.attn_config)
        self.mlp = hk.Sequential(
            [
                hk.Linear(
                    4 * self.n_embd,
                    w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
                    b_init=hk.initializers.Constant(0.0),
                    name="linear_1",
                ),
                jax.nn.gelu,
                hk.Linear(
                    self.n_embd,
                    w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
                    b_init=hk.initializers.Constant(0.0),
                    name="linear_2",
                ),
            ]
        )
        self.dropout = Dropout(self.resid_pdrop, name="dropout")

    def __call__(self, x, is_training):
        x = x + self.attn(self.ln1(x), is_training)
        x = x + self.dropout(self.mlp(self.ln2(x)), is_training=is_training)
        return x
