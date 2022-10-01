import random
from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def set_global_seed(seed: int, pytorch: bool = True) -> None:
    """
    Set the global seed for Python, NumPy, and optionally PyTorch. (NOT JAX. JAX requires explicit PRNG state.)
    """
    random.seed(seed)
    np.random.seed(seed)
    if pytorch:
        # Lazy import
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # These will degrade performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def top_k_logits(logits: jnp.array, k: int) -> jnp.array:
    """
    Given a tensor of :attr:logits, return the tensor set to -inf for non-top :attr:k logits.
    Args:
        logits (jnp.array): A tensor of shape (action_size,)
        k (int): The number of top logits to keep

    Returns:
        jnp.array: A tensor of shape (action_size,)

    Example::
    >>> logits = jnp.array([-1.0, 2.0, 1.0, 3.0])
    >>> top_k_logits(logits, k=2)
    DeviceArray([-inf,   2., -inf,   3.], dtype=float32)
    """

    # Get top k indices
    v, _ = jax.lax.top_k(logits, k)

    # Set all other logits to -inf
    return jnp.where(logits < v[-1], float("-inf"), logits)


def sample(
    params: hk.Params,
    subkey: Optional[jnp.array],
    model: Callable,
    states: jnp.array,
    block_size: int,
    temperature: float = 1.0,
    sample: bool = False,
    top_k: Optional[int] = None,
    actions: Optional[jnp.array] = None,
    rtgs: Optional[jnp.array] = None,
    timestep: Optional[jnp.array] = None,
) -> jnp.array:
    """
    Take a conditioning sequence of
    states (T, state_dim)), actions: (T, 1), rtgs: (T, 1), and timestep: (1,)
    and predict the next action token.
    Returns:
        jnp.array: A tensor of shape (1,).
    """
    # Crop context if needed
    context_length = block_size // 3
    x_cond = states if states.shape[0] <= context_length else states[-context_length:]
    if actions is not None:
        actions = actions if actions.shape[0] <= context_length else actions[-context_length:]
    if rtgs is None:
        raise ValueError("rtgs must be provided")
    rtgs = rtgs if rtgs.shape[0] <= context_length else rtgs[-context_length:]

    logits = model(params, subkey, x_cond, actions=actions, rtgs=rtgs, timestep=timestep, is_training=False)
    # Pluck the logits at the final step and scale by temperature
    logits = logits[-1, :] / temperature  # logits' shape is (action_size,)

    # Sample from the distribution or take the most likely
    if sample:
        assert subkey is not None
        # Optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = jax.nn.log_softmax(logits, axis=-1)
        ix = jax.random.categorical(subkey, probs, shape=(1,))
    else:
        # Take the most likely
        _, ix = jax.lax.top_k(logits, k=1)
    return ix


def flatten_params(params):
    for item in hk.data_structures.traverse(params):
        yield f'{item[0].replace("~/", "")}/{item[1]}', item[2]
