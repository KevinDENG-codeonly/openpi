"""JAX utilities for training Pi0.5 with real-time chunking prefixes."""

import jax.numpy as jnp

from openpi.shared import array_typing as at


@at.typecheck
def prefix_mask(delay_steps: at.Int[at.Array, " b"], horizon: int) -> at.Bool[at.Array, " b h"]:
    """Returns whether each action token belongs to the clean action prefix."""
    return jnp.arange(horizon)[None, :] < delay_steps[:, None]


@at.typecheck
def prepare_training_inputs(
    actions: at.Float[at.Array, " b h a"],
    noise: at.Float[at.Array, " b h a"],
    scalar_time: at.Float[at.Array, " b"],
    delay_steps: at.Int[at.Array, " b"],
) -> tuple[at.Float[at.Array, " b h a"], at.Float[at.Array, " b h"], at.Bool[at.Array, " b h"]]:
    """Builds flow-matching inputs with a clean RTC prefix and noisy postfix."""
    prefix = prefix_mask(delay_steps, actions.shape[1])
    token_time = jnp.where(prefix, 0, scalar_time[:, None])
    x_t = jnp.where(
        prefix[..., None],
        actions,
        token_time[..., None] * noise + (1 - token_time[..., None]) * actions,
    )
    return x_t, token_time, jnp.logical_not(prefix)


@at.typecheck
def freeze_prefix(
    x_t: at.Float[at.Array, " b h a"],
    action_prefix: at.Float[at.Array, " b h a"],
    delay_steps: at.Int[at.Array, " b"],
) -> at.Float[at.Array, " b h a"]:
    """Replaces the RTC prefix in a noisy action trajectory with clean actions."""
    prefix = prefix_mask(delay_steps, x_t.shape[1])
    return jnp.where(prefix[..., None], action_prefix, x_t)


@at.typecheck
def masked_postfix_mean(
    loss: at.Float[at.Array, " b h"], postfix_mask: at.Bool[at.Array, " b h"]
) -> at.Float[at.Array, ""]:
    """Averages each sample's postfix loss, then averages samples equally."""
    postfix_mask_f = postfix_mask.astype(loss.dtype)
    postfix_count = jnp.maximum(jnp.sum(postfix_mask_f, axis=-1), 1)
    return jnp.mean(jnp.sum(loss * postfix_mask_f, axis=-1) / postfix_count)
