from typing import Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from networks import Dropout, TransformerBlock


class GPT(hk.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        context_len: int,
        max_timestep: int,
        embd_pdrop: int,
        transformer_config: Dict,
        model_type: str,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.context_len = context_len
        self.max_block_size = context_len * 3
        self.embd_pdrop = embd_pdrop
        self.transformer_config = transformer_config
        self.max_timestep = max_timestep
        assert model_type in ["reward_conditioned", "naive"]
        self.model_type = model_type

        # Embeddings ###########################################################
        # Encode state tensor
        self.state_encoder = hk.Sequential(
            [
                hk.Conv2D(32, kernel_shape=8, stride=4, padding="VALID", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(64, kernel_shape=4, stride=2, padding="VALID", data_format="NCHW"),
                jax.nn.relu,
                hk.Conv2D(64, kernel_shape=3, stride=1, padding="VALID", data_format="NCHW"),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(
                    n_embd,
                    w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
                    b_init=hk.initializers.Constant(0.0),
                ),
                jax.nn.tanh,
            ]
        )
        # Encode rtg tensor
        self.rtg_encoder = hk.Sequential(
            [
                hk.Linear(
                    n_embd,
                    w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0),
                    b_init=hk.initializers.Constant(0.0),
                ),
                jax.nn.tanh,
            ]
        )
        # Embed ont-hot action tensor
        self.action_embeddings = hk.Sequential(
            [
                hk.Embed(vocab_size, n_embd, w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0)),
                jax.nn.tanh,
            ]
        )
        # Positional embeddings
        self.pos_emb = hk.get_parameter(
            "pos_emb", shape=[self.max_block_size + 1, n_embd], dtype=jnp.float32, init=jnp.zeros
        )  # NOTE internal(decision-transformer-jax-3)
        self.global_pos_emb = hk.get_parameter(
            "global_pos_emb", shape=[max_timestep + 1, n_embd], dtype=jnp.float32, init=jnp.zeros
        )
        ########################################################################

        # Transformer
        self.blocks = []
        for _ in range(n_layer):
            self.blocks.append(TransformerBlock(**transformer_config))
        self.dropout = Dropout(embd_pdrop)

        # Decoder head
        self.ln_f = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm_f")
        self.head = hk.Linear(
            vocab_size, with_bias=False, name="linear_head", w_init=hk.initializers.RandomNormal(stddev=0.02, mean=0.0)
        )

    def __call__(self, states, actions, rtgs, timestep, is_training=True):
        """
        Args:
            states: (T, state_dim)
            actions: (T, 1)
            rtgs: (T, 1)
            timestep: (1,) int
            is_training (bool)
        """
        T = states.shape[0]

        # Embed states
        states_emb = self.state_encoder(states.reshape((-1, 4, 84, 84)))  # (T, n_embd)
        if actions is not None and self.model_type == "reward_conditioned":
            rtgs_emb = self.rtg_encoder(rtgs)  # (T, n_embd)
            actions_emb = self.action_embeddings(actions.squeeze(-1))  # (T, n_embd)
            tokens_emb = jnp.zeros(
                (T * 3 - int(not is_training), self.n_embd)
            )  # NOTE: internal(decision-transformer-jax-1)
            tokens_emb = tokens_emb.at[::3, :].set(rtgs_emb)
            tokens_emb = tokens_emb.at[1::3, :].set(states_emb)
            tokens_emb = tokens_emb.at[2::3, :].set(actions_emb[-T + int(not is_training) :, :])
        elif (
            actions is None and self.model_type == "reward_conditioned"
        ):  # only happens at very first timestep of evaluation
            rtgs_emb = self.rtg_encoder(rtgs)  # (T, n_embd)
            tokens_emb = jnp.zeros((T * 2, self.n_embd))
            tokens_emb = tokens_emb.at[::2, :].set(rtgs_emb)
            tokens_emb = tokens_emb.at[1::2, :].set(states_emb)
        elif actions is not None and self.model_type == "naive":
            actions_emb = self.action_embeddings(actions.squeeze(-1))  # (T, n_embd)
            tokens_emb = jnp.zeros((T * 2 - int(not is_training), self.n_embd))
            tokens_emb = tokens_emb.at[::2, :].set(states_emb)
            tokens_emb = tokens_emb.at[1::2, :].set(actions_emb[-T + int(not is_training) :, :])
        elif actions is None and self.model_type == "naive":  # only happens at very first timestep of evaluation
            tokens_emb = states_emb
        else:
            raise ValueError("model_type must be 'reward_conditioned' or 'naive'")

        # Add positional embeddings
        # NOTE: internal(decision-transformer-jax-2)
        # TODO(yun-kwak): Resolve deprecation warning
        global_pos_emb = self.global_pos_emb[timestep]  # (1, n_embd)
        local_pos_emb = self.pos_emb[
            : tokens_emb.shape[0],
        ]  # (tokens_emb.shape[0], n_embd)  # NOTE: internal(decision-transformer-jax-3)
        pos_emb = global_pos_emb + local_pos_emb  # (tokens_emb.shape[0], n_embd)
        # Forward the GPT model
        x = tokens_emb + pos_emb
        x = self.dropout(x, is_training=is_training)
        for block in self.blocks:
            x = block(x, is_training=is_training)
        x = self.ln_f(x)
        logits = self.head(x)

        # Only keep logits from state_emb
        if actions is not None and self.model_type == "reward_conditioned":
            logits = logits[1::3, :]
        elif actions is None and self.model_type == "reward_conditioned":
            logits = logits[1:, :]
        elif actions is not None and self.model_type == "naive":
            logits = logits[::2, :]
        elif actions is None and self.model_type == "naive":
            logits = logits
        else:
            raise NotImplementedError()

        # (T, vocab_size)
        return logits
