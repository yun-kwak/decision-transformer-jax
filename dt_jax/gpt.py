from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from networks import Dropout, TransformerBlock


class GPT(hk.Module):
    def __init__(self, vocab_size, n_embd, n_layer, block_size, embd_pdrop, transformer_config, name):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.block_size = block_size
        self.embd_pdrop = embd_pdrop
        self.transformer_config = transformer_config

        # input embedding stem
        self.dropout = Dropout(embd_pdrop)
        self.tok_emb = hk.Embed(vocab_size, n_embd, name="tok_emb")
        self.pos_emb = hk.get_parameter("embeddings", shape=[block_size, n_embd], dtype=jnp.float32, init=jnp.zeros)

        self.blk_fn = partial(
            TransformerBlock,
            n_embd=transformer_config["n_embd"],
            attn_config=transformer_config["attn_config"],
            resid_pdrop=transformer_config["resid_pdrop"],
        )
        self.blocks = []
        for _ in range(self.n_layer):
            self.blocks.append(self.blk_fn())
        # decoder head
        self.ln_f = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm_f")
        self.head = hk.Linear(vocab_size, with_bias=False, name="linear_head")

    # TODO: Implement configure_optimizers, _init_weights methods
    def _configure_optimizers(self):
        pass

    def _init_weights(self):
        pass

    def __call__(self, x, actions, targets, rtgs, timesteps, is_training=True):
        t = x.shape[0]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(x)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:t, :]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        x = self.dropout(x, is_training=is_training)
        # transformer
        for block in self.blocks:
            x = block(x, is_training=is_training)
        x = self.ln_f(x)
        return self.head(x)


def cross_entropy(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jax.nn.log_softmax(logits) * one_hot
    loss = loss.sum() / one_hot.sum()
    return loss


def loss_fn(func, idx, targets, config, is_training):
    return cross_entropy(jax.vmap(func, in_axes=[0, None, None])(idx, config, is_training), targets)
